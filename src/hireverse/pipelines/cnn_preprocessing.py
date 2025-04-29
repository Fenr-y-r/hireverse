import gc
from multiprocessing import Pool, cpu_count
import random
import shutil
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
import copy


def get_processed_frames_images(vid_file_path: str, participant_id):
    face_analyzer = FaceAnalyzer()
    no_frames = face_analyzer.get_video_frame_count(vid_file_path)
    # try:
    for frame in face_analyzer.yield_video_frames(
        vid_file_path, participant_id
    ):
        try:
            landmarks_obj = face_analyzer.process_image_results(frame.image)
            if not landmarks_obj:
                continue  # Skip frames without detectable faces
            original_frame = copy.deepcopy(frame)
            frame.facial_landmarks = landmarks_obj.landmark

            if frame.image.shape[1] != 640:
                frame.resize(new_width=640)

            frame.align_face_with_mediapipe_landmarks()

            frame.face = face_analyzer.get_face_coordinates(
                frame.facial_landmarks, frame.image
            )
            x, y, w, h = frame.face
            cropped_face_image = frame.get_cropped_image(x, y, x + w, y + h)
            if cropped_face_image is None:
                continue
            frame.image = cropped_face_image

            frame.image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            # normalized = np.clip(frame.image.astype("float32") / 255.0, 0.0, 1.0)

            frame.resize(new_width=640, new_height=640)

            yield frame.image

        finally:
            if "landmarks_obj" in locals():
                del landmarks_obj
            del frame

    # except Exception as e:
    #     print(f"Error processing {vid_file_path}: {str(e)}")
    #     raise
    # finally:
    #     del face_analyzer


def show_image(frame):
    frame.reset_drawable_image()
    # frame.draw_face_border()
    # frame.draw_facial_landmarks()
    # if frame.facial_landmarks:
    #     frame.draw_circle_at_facial_landmark(frame.facial_landmarks[10], frame.facial_landmarks[152])
    frame.display()


def process_single_video(vid_path, output_dir, participant_id):
    labels = get_labels_dict(participant_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for idx, img in enumerate(get_processed_frames_images(vid_path, participant_id)):
        output_path = os.path.join(output_dir, f"frame{idx}.jpg")
        cv2.imwrite(
            output_path,
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],  # Adjust quality (1-100)
        )
        del img  # optional, helps if image is large

    print(f"Saved {participant_id}")


def process_single_video_wrapper(args):
    """Wrapper function for multiprocessing"""
    vid_path, output_dir, participant_id = args
    try:
        process_single_video(vid_path, output_dir, participant_id)
        return True
    except Exception as e:
        print(f"Error processing {participant_id}: {str(e)}")
        return False


if __name__ == "__main__":
    video_dir = os.path.join(BASE_DIR, "data", "raw", "videos")
    output_dir = os.path.join(BASE_DIR, "data", "processed", "videos_frames")
    participant_ids = get_participant_ids()

    inputs = []
    for participant_id in participant_ids:
        input_video_path = os.path.join(video_dir, f"{participant_id}.avi")
        participant_dir = os.path.join(output_dir, participant_id)
        inputs.append((input_video_path, participant_dir, participant_id))

    max_workers = cpu_count() - 1
    with Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_video_wrapper, inputs),
                total=len(inputs),
                desc="Processing videos",
            )
        )
