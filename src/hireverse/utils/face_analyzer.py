import math
import time
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Set, Tuple
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmark,
    NormalizedLandmarkList,
)
import os
import re
import cv2
import random
from deepface import DeepFace
import pandas as pd
import sys
from pathlib import Path
from hireverse.schemas.frame import Frame as fa
from hireverse.schemas.selected_facial_landmarks import TwoLandmarksConnector
from typing import Optional


class FaceAnalyzer:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, min_detection_confidence=0.5, max_num_faces=1
    )
    VIDEOS_FOLDER_PATH = "./MIT/Videos/"

    def _calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def align_face(self, frame, landmarks):
        """
        Aligns the face using MediaPipe landmarks.

        Args:
            frame (numpy.ndarray): Input frame (RGB format).
            landmarks: MediaPipe face landmarks.

        Returns:
            numpy.ndarray: Aligned face image.
        """
        # Get eye landmarks (left and right)
        left_eye = np.array(
            [landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0]]
        )
        right_eye = np.array(
            [landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0]]
        )

        # Calculate the angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Rotate the frame to align the eyes horizontally
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_frame = cv2.warpAffine(
            frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC
        )

        return aligned_frame

    def get_smile_from_frame(self, face_roi):
        """
        Measures happiness (smile) intensity in a frame using DeepFace's emotion model.

        Args:
            frame (numpy.ndarray): Input frame (BGR format).

        Returns:
            float: Happiness probability (0-1).
        """
        if face_roi is None:
            return None
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        # TODO: align face for better accuracy
        result = DeepFace.analyze(
            face_roi,
            align=False,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
        )
        # happiness_prob = result[0]["emotion"]["happy"] / 100  # Normalize to [0, 1]
        return round(result[0]["emotion"]["happy"])

    def get_face_roi_image(self, image, face, expand_ratio=1.0):
        """
        Extracts the face region of interest (ROI) from the image.

        Args:
            image (numpy.ndarray): Input image.
            face (tuple): Face bounding box (x, y, w, h).
            expand_ratio (float): Ratio to expand the bounding box (default: 1.0).

        Returns:
            numpy.ndarray: Cropped face ROI.
        """
        if not face:
            return None

        x, y, w, h = face

        # Calculate expanded bounding box
        new_w = w * expand_ratio
        new_h = h * expand_ratio
        new_x = x - (new_w - w) / 2
        new_y = y - (new_h - h) / 2

        # Ensure the expanded bounding box stays within the image boundaries
        new_x = max(0, int(new_x))
        new_y = max(0, int(new_y))
        new_w = min(image.shape[1] - new_x, int(new_w))
        new_h = min(image.shape[0] - new_y, int(new_h))

        # Extract the face ROI
        face_roi = image[new_y : new_y + new_h, new_x : new_x + new_w]
        return face_roi

    def get_face_coordinates(self, face_landmarks: list[NormalizedLandmark], image):
        """
        This method returns the coordinates of the face in the form of a tuple (x, y, w, h).
        """

        # Calculate the bounding box for each face
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]
        # TODO: replace the follwoing with   bboxC = detection.location_data.relative_bounding_box
        # Calculate the bounding box coordinates (normalized)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        img_h = image.shape[1]
        img_w = image.shape[0]
        return (
            int(min_x * img_h),
            int(min_y * img_w),
            round((max_x - min_x) * img_h),
            round((max_y - min_y) * img_w),
        )

    def get_bouding_box_center(self, bbox):
        """
        Extract the center of a single bounding box from its coordinates.

        Args:
            bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box.

        Returns:
            Tuple: (center_x, center_y) representing the center of the bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return (center_x, center_y)

    def get_displacement_between_two_bounding_boxes(self, bbox1, bbox2):
        """
        Returns the Euclidean distance (displacement) between two bounding box centers.
        """
        delta = np.array(bbox2) - np.array(bbox1)
        return np.linalg.norm(delta)

    def get_horizontal_distance_between_two_bounding_boxes(self, bbox1, bbox2):
        """
        Returns the absolute left-right movement between two bounding boxes.
        """
        delta = np.array(bbox2) - np.array(bbox1)
        return abs(delta[0])

    def get_vertical_displacement_between_two_bounding_boxes(self, bbox1, bbox2):
        """
        Returns the absolute up-down movement between two bounding boxes.
        """
        delta = np.array(bbox2) - np.array(bbox1)
        return abs(delta[1])

    def process_image_results(self, image) -> Optional[NormalizedLandmarkList]:
        image.flags.writeable = False
        results = FaceAnalyzer.face_mesh.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).multi_face_landmarks
        image.flags.writeable = True
        return results[0] if results else None

    # TODO: make sure this works
    def get_largest_face_landmarks_obj(self, image, detected_faces_landmarks):
        """
        This function takes the MediaPipe results and returns the largest face landmarks
        based on bounding box area.
        """
        if detected_faces_landmarks:
            max_area = 0
            largest_face_landmarks = None

            for face_landmarks_obj in detected_faces_landmarks:
                _, _, w, h = self.get_face_coordinates(
                    face_landmarks_obj.landmark, image
                )
                area = w * h

                if area > max_area:
                    max_area = area
                    largest_face_landmarks = face_landmarks_obj
            return largest_face_landmarks

        return None

    # fmt: off
    def _get_brow_interest_points(self, face_interest_points):
        outer_brow_left = face_interest_points[276]
        inner_brow_left = face_interest_points[285]

        inner_brow_right = face_interest_points[55]
        outer_brow_right = face_interest_points[46]
        return (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right)

    def _get_eye_interest_points(self, face_interest_points: List[Tuple[int, int]]):
        beside_left_eye_outer = face_interest_points[446]  # change name
        left_eye_inner = face_interest_points[362]
        right_eye_inner = face_interest_points[133]
        beside_right_eye_outer = face_interest_points[35]  # change
        
        right_eye_upper =face_interest_points[159]
        right_eye_lower =face_interest_points[145]
        left_eye_upper = face_interest_points[386]
        left_eye_lower = face_interest_points[374]
        
        return (beside_left_eye_outer, beside_right_eye_outer, left_eye_inner, right_eye_inner, right_eye_upper, right_eye_lower, left_eye_upper, left_eye_lower)

    def get_lips_coordinates(self, face_interest_points: List[Tuple[int, int]]):
        outer_upper_lip = face_interest_points[0]
        inner_upper_lip = face_interest_points[13]
        inner_lower_lip = face_interest_points[14]
        outer_lower_lip = face_interest_points[17]
        lip_corner_right = face_interest_points[291]
        lip_corner_left = face_interest_points[61]
        return (
            outer_upper_lip,
            outer_lower_lip,
            inner_upper_lip,
            inner_lower_lip,
            lip_corner_right,
            lip_corner_left,
        )

    def get_selected_facial_landmarks(self, face_interest_points: list[Tuple[int, int, int]]) -> list[TwoLandmarksConnector]:
        selected_facial_landmarks =[]
        if face_interest_points:                
            (outer_lip_above,outer_lip_below,inner_lip_above,inner_lip_below,lip_corner_right,lip_corner_left,) = self.get_lips_coordinates(face_interest_points)
            (beside_left_eye_outer, beside_right_eye_outer, left_eye_inner, right_eye_inner, right_eye_upper, right_eye_lower, left_eye_upper, left_eye_lower) = (self._get_eye_interest_points(face_interest_points))
            (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right) = self._get_brow_interest_points(face_interest_points)

            selected_facial_landmarks.append(TwoLandmarksConnector("average_outer_brow_height", (outer_brow_left, beside_left_eye_outer, outer_brow_right, beside_right_eye_outer)))
            selected_facial_landmarks.append(TwoLandmarksConnector("average_inner_brow_height", (inner_brow_left, left_eye_inner, inner_brow_right, right_eye_inner)))
            selected_facial_landmarks.append(TwoLandmarksConnector("eye_open", (right_eye_upper, right_eye_lower, left_eye_upper, left_eye_lower)))
            selected_facial_landmarks.append(TwoLandmarksConnector("outer_lip_height", (outer_lip_above, outer_lip_below)))
            selected_facial_landmarks.append(TwoLandmarksConnector("inner_lip_height", (inner_lip_above, inner_lip_below)))
            selected_facial_landmarks.append(TwoLandmarksConnector("lip_corner_distance", (lip_corner_left, lip_corner_right)))

            return selected_facial_landmarks
        return None
    # fmt: on

    def get_face_angles(
        self, image, face_landmarks: list[NormalizedLandmark], isWebcam=False
    ):
        def _rotation_matrix_to_angles(rotation_matrix):
            """
            returns a rounded tuple (x, y, z) representing the angles of the face in degrees.
            """
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(
                -rotation_matrix[2, 0],
                math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
            )
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            return np.array([x, y, z]) * 180.0 / math.pi

        if face_landmarks is None:
            return None
        face_coordination_in_real_world = np.array(
            [
                [285, 528, 200],
                [285, 371, 152],
                [197, 574, 128],
                [173, 425, 108],
                [360, 574, 128],
                [391, 425, 108],
            ],
            dtype=np.float64,
        )

        h, w, _ = image.shape
        face_coordination_in_image = []

        for idx, lm in enumerate(face_landmarks):
            if idx in [1, 9, 57, 130, 287, 359]:
                x, y = int(lm.x * w), int(lm.y * h)
                face_coordination_in_image.append([x, y])

        face_coordination_in_image = np.array(
            face_coordination_in_image, dtype=np.float64
        )

        focal_length = 1 * w
        cam_matrix = np.array(
            [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]]
        )
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, transition_vec = cv2.solvePnP(
            face_coordination_in_real_world,
            face_coordination_in_image,
            cam_matrix,
            dist_matrix,
        )

        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

        result = _rotation_matrix_to_angles(rotation_matrix)
        if not isWebcam:
            add = [2, 27, 0]
        else:
            add = [0, 0, 0]

        for i in range(len(result)):
            result[i] = result[i] + add[i]

        return tuple(result)

    def get_one_frame_per_video(self) -> List[fa]:
        frames = []
        files = sorted(os.listdir(FaceAnalyzer.VIDEOS_FOLDER_PATH))
        files.sort(key=lambda f: int(re.search(r"\d+", f).group()))
        for f in files:
            if f.endswith(".avi") and not f.startswith("PP"):
                file_path = os.path.join(FaceAnalyzer.VIDEOS_FOLDER_PATH, f)
                cap = cv2.VideoCapture(file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                middle_frame = random.randint(total_frames // 4, 3 * total_frames // 4)
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES, middle_frame
                )  #  move the pointer to the desired frame
                success, frame_image = (
                    cap.read()
                )  # grabs the frame at the current position.
                cap.release()
                participant_number = re.search(r"\d+", f).group()
                frames.append(
                    fa(
                        participant_number,
                        participant_number,
                        frame_image,
                        is_categorized_by_participant=True,
                    )
                )

        return frames

    def get_folder_path(self, participant_id, video_folder_path: str):
        return os.path.join(video_folder_path, f"{participant_id}.avi")

    def get_video_frames_for_participant(
        self,
        participant_id: str,
        video_folder_path: str,
        num_selected_frames: int = None,
        is_random=True,
        target_fps=None,
    ) -> List[fa]:
        """
        Retrieves video frames for a specific participant from the given video folder path.
        Args:
            participant_id (str): The unique identifier for the participant.
            video_folder_path (str): The path to the folder containing the participant's video.
            num_selected_frames (int, optional): The number of frames to select from the video.
                If specified, the frames will be selected consecutively unless `is_random` is set to True.
            is_random (bool, optional): Whether to select frames randomly. Defaults to True. Only used if num_selected_frames is specified.
            target_fps (float, optional): The target frames per second for the video.
        Returns:
            List[Frame]: A list of frames extracted from the video.
        """

        video_path = self.get_folder_path(participant_id, video_folder_path)
        return self._get_video_frames(
            video_path,
            participant_id,
            num_selected_frames,
            is_random=is_random,
            target_fps=target_fps,
        )

    def get_video_frames(
        self,
        video_path,
        participant_id,
        num_selected_frames: int = None,
        is_consecutive=True,
        target_fps=None,
    ) -> List[fa]:
        return self._get_video_frames(
            video_path,
            participant_id=participant_id,
            num_selected_frames=num_selected_frames,
            is_random=is_consecutive,
            target_fps=target_fps,
        )

    # TODO: num_selected_frames and is_random are not implrmeneted yet
    def _get_video_frames(
        self,
        video_path: str,
        participant_id: int,
        num_selected_frames: Optional[int],
        is_random: bool,
        target_fps: Optional[int],
    ) -> List[fa]:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame_image = cap.read()
            if ret:
                frames.append(
                    fa(
                        i,
                        participant_id if participant_id is not None else 0,
                        frame_image,
                    )
                )

        cap.release()

        if target_fps:
            frames = self._adjust_frames_list_acc_to_fps(
                frames, original_fps, target_fps
            )

        return frames[:num_selected_frames] if num_selected_frames else frames

    def get_video_frame_count(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        lol = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return lol

    def yield_video_frames(
        self, video_path: str, participant_id: int, num_selected_frames: float = None
    ):
        cap = cv2.VideoCapture(video_path)
        if not num_selected_frames:
            num_selected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_index = 0
        while True:
            ret, frame_image = cap.read()
            if not ret:
                break
            yield fa(frame_index, participant_id or 0, frame_image)
            if frame_index == num_selected_frames:
                break
            frame_index += 1

        cap.release()

    def _adjust_frames_list_acc_to_fps(
        self, original_frames: List[np.ndarray], original_fps: float, target_fps
    ) -> List[np.ndarray]:
        step = original_fps / target_fps
        # fmt: off
        return [original_frames[int(i * step)] for i in range(int(len(original_frames) / step))]    

    def display_image(self, image, title=None):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
