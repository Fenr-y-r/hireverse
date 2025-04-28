import random
import uuid
import cv2
from hireverse.utils.utils import *
from hireverse.utils.face_analyzer import FaceAnalyzer
from hireverse.schemas.frame import Frame
from typing import List


def get_processed_frames( vid_file_path: str):
    face_analyzer = FaceAnalyzer()

    frames = face_analyzer.get_video_frames(
        participant_id = str(uuid.uuid4()),
        video_path=vid_file_path,
        target_fps=20,
    )

    filtered_frames: List[Frame] = []
    for frame in frames:
        detected_faces_landmarks = face_analyzer.process_image_results(frame.image)
        frame.facial_landmarks_obj = face_analyzer.get_largest_face_landmarks_obj(
            frame.image, detected_faces_landmarks
        )
        if frame.facial_landmarks_obj:
            frame.facial_landmarks = frame.facial_landmarks_obj.landmark
            filtered_frames.append(frame)
    frames = filtered_frames

    for frame in frames:
        frame.resize(new_width=640)
        frame.align_face_with_mediapipe_landmarks()

        if frame.facial_landmarks:
            frame.face = face_analyzer.get_face_coordinates(
                frame.facial_landmarks, frame.image
            )
            x, y, w, h = frame.face
            frame.crop_frame(x, y, x + w, y + h)

        frame.image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        frame.image = frame.image.astype("float32") / 255.0
        frame.resize(new_width=640, new_height=640)

    return frames
