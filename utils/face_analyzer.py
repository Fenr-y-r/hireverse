import math
import time
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import os
import re
import cv2
import random

from models.frame import Frame
from models.selected_facial_landmarks import SelectedFacialLandmarks
from utils.utils import denormalize_int

class FaceAnalyzer:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, min_detection_confidence=0.5
    )        
    FOLDER_PATH = "./datasets/MIT/Videos/"

    def _calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def get_smile(self, frame_image, face):
        (x, y, w, h) = face
        gray_image = self.convert_to_gray(frame_image)
        roi_gray = gray_image[y : y + h, x : x + w]
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.04,  # Compensates for that an object at different distances from the camera will appear at different sizes. A lower scaleFactor increases the detection time but also increases the chance of detection.  # Typical values range from 1.01 to 1.3.
            minNeighbors=47,  # Higher values result in fewer detections but with higher quality. Lower values may lead to more detections but with possible false positives. It’s a trade-off between precision and recall.
            # minSize=(30, 30)    # smiles smaller than this size are ignored.  # TODO: change according to the face distance from webcam
        )
        if len(smiles):
            largest_smile = max(smiles, key=lambda s: s[2] * s[3])
            (sx, sy, sw, sh) = largest_smile
            if sy > h // 2:
                return (x + sx, y + sy, sw, sh)

        return None

    def get_face_coordinates(
        self, face_landmarks: list[NormalizedLandmark] , image
    ):
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


    def process_image_results (self,image) :
        image.flags.writeable = False
        results = FaceAnalyzer.face_mesh.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).multi_face_landmarks
        image.flags.writeable = True
        return results

    # TODO: make sure this works
    def get_largest_face_landmarks_obj(self, image, detected_faces_landmarks) -> list[NormalizedLandmark]:
        """
        This function takes the MediaPipe results and returns the largest face landmarks
        based on bounding box area.
        """

        if detected_faces_landmarks:
            max_area = 0
            largest_face_landmarks = None

            for face_landmarks_obj in detected_faces_landmarks:
                _, _, w, h = self.get_face_coordinates(face_landmarks_obj.landmark, image)
                area = w * h

                if area > max_area:
                    max_area = area
                    largest_face_landmarks = face_landmarks_obj
            return largest_face_landmarks

        return None

    def _get_brow_interest_points(
        self, face_interest_points
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        outer_brow_left = face_interest_points[276]
        inner_brow_left = face_interest_points[285]

        inner_brow_right = face_interest_points[55]
        outer_brow_right = face_interest_points[46]
        return (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right)

    def _get_eye_interest_points(
        self, face_interest_points: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        eye_outer_left = face_interest_points[33]
        eye_inner_left = face_interest_points[133]
        eye_inner_right = face_interest_points[362]
        eye_outer_right = face_interest_points[263]
        return (eye_outer_left, eye_outer_right, eye_inner_left, eye_inner_right)

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

    def _euclidean_distance_between_two_interest_points(self, point1, point2):
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

    def _get_lip_lengths(self, face_interest_points: List[Tuple[int, int]]):
        (
            outer_lip_above,
            outer_lip_below,
            inner_lip_above,
            inner_lip_below,
            lip_corner_right,
            lip_corner_left,
        ) = self.get_lips_coordinates(face_interest_points)
        outer_lip_height = self._euclidean_distance_between_two_interest_points(
            outer_lip_above, outer_lip_below
        )
        inner_lip_height = self._euclidean_distance_between_two_interest_points(
            inner_lip_above, inner_lip_below
        )
        lip_corner_distance = self._euclidean_distance_between_two_interest_points(
            lip_corner_left, lip_corner_right
        )
        return (outer_lip_height, inner_lip_height, lip_corner_distance)

    def get_selected_facial_landmarks(
        self, face_interest_points: List[Tuple[int, int, int]]
    ) -> SelectedFacialLandmarks:
        if face_interest_points:
            (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right) = (self._get_brow_interest_points(face_interest_points))
            (eye_outer_left, eye_outer_right, eye_inner_left, eye_inner_right) = (
                self._get_eye_interest_points(face_interest_points)
            )
            (outer_lip_height, inner_lip_height, lip_corner_distance) = (
                self._get_lip_lengths(face_interest_points)
            )
            (
                outer_lip_above,
                outer_lip_below,
                inner_lip_above,
                inner_lip_below,
                lip_corner_right,
                lip_corner_left,
            ) = self.get_lips_coordinates(face_interest_points)
            return SelectedFacialLandmarks(
                inner_brow_left=inner_brow_left,
                outer_brow_left=outer_brow_left,
                inner_brow_right=inner_brow_right,
                outer_brow_right=outer_brow_right,
                eye_outer_left=eye_outer_left,
                eye_outer_right=eye_outer_right,
                eye_inner_left=eye_inner_left,
                eye_inner_right=eye_inner_right,
                outer_lip_height=outer_lip_height,
                inner_lip_height=inner_lip_height,
                lip_corner_distance=lip_corner_distance,
                outer_lip_above=outer_lip_above,
                outer_lip_below=outer_lip_below,
                inner_lip_above=inner_lip_above,
                inner_lip_below=inner_lip_below,
                lip_corner_right=lip_corner_right,
                lip_corner_left=lip_corner_left,
            )
        return None
    
    def get_face_angles(self, image, face_landmarks: list[NormalizedLandmark], isWebcam=False):
        def _rotation_matrix_to_angles(rotation_matrix):
            """
            returns a rounded tuple (x, y, z) representing the angles of the face in degrees.
            """
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                            rotation_matrix[1, 0] ** 2))
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            return np.array([x, y, z]) * 180. / math.pi

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = image.shape
        face_coordination_in_image = []
        
        for idx, lm in enumerate(face_landmarks):
            if idx in [1, 9, 57, 130, 287, 359]:
                x, y = int(lm.x * w), int(lm.y * h)
                face_coordination_in_image.append([x, y])

        face_coordination_in_image = np.array(face_coordination_in_image,
                                            dtype=np.float64)

        focal_length = 1 * w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                            [0, focal_length, h / 2],
                            [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, transition_vec = cv2.solvePnP(
            face_coordination_in_real_world, face_coordination_in_image,
            cam_matrix, dist_matrix)

        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

        result = _rotation_matrix_to_angles(rotation_matrix)
        if not isWebcam:
            add = [2,27,0]
        else:
            add = [0,0,0]
        
        for i  in range(len(result)):
            result[i]= result[i]+ add[i]
        
        
        return tuple(result)

    
    def get_one_frame_per_video(self) -> List[Frame]:
        frames = []
        files = sorted(os.listdir(FaceAnalyzer.FOLDER_PATH))
        files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
        for f in files:
            if f.endswith(".avi") and not f.startswith("PP"):
                file_path = os.path.join(FaceAnalyzer.FOLDER_PATH, f) 
                cap = cv2.VideoCapture(file_path)  
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
                middle_frame = random.randint(total_frames // 4, 3 * total_frames // 4)  
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)  #  move the pointer to the desired frame 
                success, frame_image = cap.read() # grabs the frame at the current position.
                cap.release() 
                participant_number = re.search(r'\d+', f).group()
                frames.append(Frame(participant_number, participant_number, frame_image,is_categorized_by_participant=True))  
                
        return frames
    
    def get_video_frames_for_participant(self, participant_number: str, num_selected_frames :int=None) -> List[Frame]:
        video_path = os.path.join(FaceAnalyzer.FOLDER_PATH, f"P{participant_number}.avi")
        return self._get_video_frames(video_path,  participant_number , num_selected_frames)
    
    def get_video_frames(self, video_path , num_selected_frames :int=None) -> List[Frame]:
        return self._get_video_frames(video_path, num_selected_frames= num_selected_frames)
    
    def _get_video_frames(self, video_path, participant_number=None,  num_selected_frames :int=None ,) -> List[Frame]:
        frames: List[Frame] = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = range(frame_count)
        if num_selected_frames:
            indices = sorted(random.sample(range(frame_count), num_selected_frames)) 
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index) 
            ret, frame_image = cap.read()
            frames.append(Frame(len(frames), participant_number if participant_number is not None else 0, frame_image))
        cap.release()
        return frames


  