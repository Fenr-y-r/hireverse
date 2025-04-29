import random
import time
import uuid
import cv2
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
    vid_name = Path(vid_path).stem
    output_path = os.path.join(output_dir, f"{vid_name}.npz")
    labels = get_labels_dict_from_participant_id(vid_name)
    np.savez(output_path, frames=frame_data, **labels)
    print(f"\nDone! Saved {vid_name}")
    return True
  


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