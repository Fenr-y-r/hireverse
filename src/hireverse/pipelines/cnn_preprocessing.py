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


def get_processed_frames_images(vid_file_path: str, participant_id):
    face_analyzer = FaceAnalyzer()
    no_frames   = face_analyzer.get_video_frame_count(vid_file_path)
    try:
        for frame in tqdm(face_analyzer.yield_video_frames(
            vid_file_path, participant_id, target_fps=20
        ), total=no_frames):
            try:
                landmarks_obj = face_analyzer.process_image_results(frame.image)
                if not landmarks_obj:
                    continue  # Skip frames without detectable faces

                frame.facial_landmarks = landmarks_obj.landmark

                if frame.image.shape[1] != 640:
                    frame.resize(new_width=640)

                frame.align_face_with_mediapipe_landmarks()

                x, y, w, h = face_analyzer.get_face_coordinates(
                    frame.facial_landmarks, frame.image
                )
                frame.crop_frame(x, y, x + w, y + h)

                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

                normalized = np.clip(gray.astype("float32") / 255.0, 0.0, 1.0)

                final_img = cv2.resize(normalized, (640, 640))

                yield final_img

            finally:
                if "landmarks_obj" in locals():
                    del landmarks_obj
                del frame
                gc.collect()

    except Exception as e:
        print(f"Error processing {vid_file_path}: {str(e)}")
        raise
    finally:
        del face_analyzer
    # for frame in frames[:7]:
    #     frame.reset_drawable_image()
    #     # frame.draw_face_border()
    #     # frame.draw_facial_landmarks()
    #     # if frame.facial_landmarks:
    #     #     frame.draw_circle_at_facial_landmark(frame.facial_landmarks[10], frame.facial_landmarks[152])
    #     frame.display()


def process_single_video(vid_path, output_dir):
    participant_id = Path(vid_path).stem
    output_path = os.path.join(output_dir, f"{participant_id}.h5")
    labels = get_labels_dict_from_participant_id(participant_id)

    with h5py.File(output_path, "w") as h5f:
        dset = h5f.create_dataset(
            "frames",
            shape=(0, 640, 640),
            maxshape=(None, 640, 640),
            dtype="float32",
            chunks=(1, 640, 640),
            compression="gzip",
        )
        lbl_grp = h5f.create_group("labels")
        for k, v in labels.items():
            lbl_grp.attrs[k] = v

        for idx, img in enumerate(
            get_processed_frames_images(vid_path, participant_id)
        ):
            dset.resize((idx + 1, 640, 640))
            dset[idx] = img
            del img  # optional, helps if image is large

    print(f"Saved {participant_id}")


def get_labels_dict_from_participant_id(participant_id: str):
    df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "external", "turker_scores_full_interview.csv"),
    )
    lol = df.loc[
        (df["Participant"] == participant_id.lower()) & (df["Worker"] == "AGGR")
    ]
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
