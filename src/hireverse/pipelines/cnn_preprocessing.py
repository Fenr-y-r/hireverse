import gc
import random
import time
import uuid
import cv2
import h5py
from hireverse.utils.utils import *
from hireverse.utils.face_analyzer import FaceAnalyzer
from hireverse.schemas.frame import Frame
from typing import List
import concurrent.futures
import os
import pickle
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from pathlib import Path
import numpy as np


def get_processed_frames_images(vid_file_path: str):

    face_analyzer = FaceAnalyzer()
    frames = face_analyzer.get_video_frames(
        participant_id="randomId", video_path=vid_file_path, target_fps=20
    )
    filtered_frames: List[Frame] = []
    for frame in frames:
        frame.facial_landmarks_obj = face_analyzer.process_image_results(frame.image)
        if frame.facial_landmarks_obj:
            frame.facial_landmarks = frame.facial_landmarks_obj.landmark
            filtered_frames.append(frame)
    frames = filtered_frames

    for frame in frames:
        new_width = 640
        _, width = frame.image.shape[:2]
        if width != new_width:
            frame.resize(new_width=new_width)

        frame.align_face_with_mediapipe_landmarks()

        frame.face = face_analyzer.get_face_coordinates(
            frame.facial_landmarks, frame.image
        )
        x, y, w, h = frame.face
        frame.crop_frame(x, y, x + w, y + h)

        frame.image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        frame.image = np.clip(frame.image.astype("float32") / 255.0, 0.0, 1.0)

        frame.resize(new_width=640, new_height=640)

    # for frame in frames[:7]:
    #     frame.reset_drawable_image()
    #     # frame.draw_face_border()
    #     # frame.draw_facial_landmarks()
    #     # if frame.facial_landmarks:
    #     #     frame.draw_circle_at_facial_landmark(frame.facial_landmarks[10], frame.facial_landmarks[152])
    #     frame.display()

    return [frame.image for frame in frames]


def process_single_video(vid_path, output_dir):
    frame_data = get_processed_frames_images(vid_path)
    participant_id = Path(vid_path).stem
    output_path = os.path.join(output_dir, f"{participant_id}.h5")
    labels = get_labels_dict_from_participant_id(participant_id)
    with h5py.File(output_path, "w") as h5f:
        # Create a resizable dataset: 0 initial frames, unlimited max
        dset = h5f.create_dataset(
            "frames",
            shape=(0, 640, 640),    # starts with zero frames stored
            maxshape=(None, 640, 640),  # unlimited frames
            dtype="float32",
            chunks=(1, 640, 640),   # 1 frame of size 640*640 at a time 
            compression="gzip"
        )
        # Store labels as attributes in a labels group
        lbl_grp = h5f.create_group("labels")
        for k, v in labels.items():
            lbl_grp.attrs[k] = v

        # Append each frame one by one
        for idx, img in enumerate(frame_data):
            dset.resize((idx + 1, 640, 640))    # Each loop increases the dataset length by one
            # write the new slice
            dset[idx, :, :] = img
        del frame_data; gc.collect()    # garbage collector frees memory by removing objects that your program no longer uses

    print(f"Saved {participant_id}")
  


def get_labels_dict_from_participant_id(participant_id: str):
    df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "external", "turker_scores_full_interview.csv"),
    )
    lol = df.loc[(df["Participant"] == participant_id.lower()) & (df["Worker"] == "AGGR")]
    return lol.iloc[0].to_dict()
    



if __name__ == "__main__":
    video_dir = os.path.join(BASE_DIR, "data", "raw", "videos")
    output_dir = os.path.join(BASE_DIR, "data", "processed", "npz_frames")

    participant_ids = get_participant_ids()
    video_paths = [
        os.path.join(video_dir, f"{participant_id}.avi")
        for participant_id in participant_ids
    ]
    for video_path in video_paths:
        process_single_video(video_path, output_dir)