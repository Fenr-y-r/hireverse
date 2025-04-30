import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import (
    NormalizedLandmark,
    NormalizedLandmarkList,
)
from typing import List, Tuple
import numpy as np
from hireverse.schemas.selected_facial_landmarks import (
    TwoLandmarksConnector,
)
import os
from hireverse.utils.utils import denormalize_landmarks_without_Z
import hireverse.utils.face_analyzer as fa


class Frame:

    def __init__(
        self,
        index: int,
        participant: str,
        image: np.ndarray,
        is_categorized_by_participant: bool = False,
    ):
        self.index = index
        self.participant_name = participant
        self.image = image
        self.smile_area = None
        self.face: Tuple[int, int, int, int] = None
        self.smile: Tuple[int, int, int, int] = None
        self.facial_landmarks: NormalizedLandmarkList = None
        self.copied_image_for_drawing = None
        self.two_landmarks_connectors: List[TwoLandmarksConnector]
        self.image = image
        self.faces = None
        self.facial_landmarks_obj = None
        self.face_angles: Tuple[int, int, int] = None
        self.is_categorized_by_participant = is_categorized_by_participant
        self.head_displacement = None
        self.head_vertical_displacement = None
        self.head_horizontal_displacement = None

    def _create_drawable_image_copy_if_not_exist(self):
        if self.copied_image_for_drawing is None:
            self.copied_image_for_drawing = self.image.copy()

    def draw_smile(self):
        self._create_drawable_image_copy_if_not_exist()
        if self.smile:
            self.draw_rectangle(self.smile, (0, 255, 0))

    def draw_nose_line(self, p1, p2):
        self._create_drawable_image_copy_if_not_exist()

    def draw_face_border(self):
        self._create_drawable_image_copy_if_not_exist()
        if self.face is not None:
            self.draw_rectangle(self.face, (255, 0, 0))

    def draw_facial_landmarks(self):
        self._create_drawable_image_copy_if_not_exist()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        if self.facial_landmarks:
            mp_drawing.draw_landmarks(
                self.copied_image_for_drawing,
                self.facial_landmarks_obj,
                None,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 0), thickness=0, circle_radius=0
                ),
                connection_drawing_spec=None,
            )

    def put_face_angles(self):
        if self.face_angles:
            for i, angle_name in enumerate("XYZ"):
                self.put_text(
                    f"{angle_name}: {round(self.face_angles[i], 1)}", (20, 20 + i * 20)
                )

    def draw_circle_at_facial_landmark(self, *args: NormalizedLandmark):
        for landmark in args:
            self.draw_cirle_by_coordinate(
                denormalize_landmarks_without_Z(landmark, self.image)
            )

    def draw_selected_facial_landmarks(self, draw_lines=True):
        self._create_drawable_image_copy_if_not_exist()
        if self.two_landmarks_connectors:
            for two_landmarks_connector in self.two_landmarks_connectors:
                if draw_lines:
                    color = tuple(np.random.randint(0, 256, 3).tolist())
                    producing_landmarks = two_landmarks_connector.producing_landmarks
                    for i in range(0, len(producing_landmarks), 2):
                        self.draw_line(
                            denormalize_landmarks_without_Z(
                                producing_landmarks[i],
                                self.image,
                            ),
                            denormalize_landmarks_without_Z(
                                producing_landmarks[i + 1],
                                self.image,
                            ),
                            color=color,
                        )
                        self.draw_circle_at_facial_landmark(producing_landmarks[i])
                        self.draw_circle_at_facial_landmark(producing_landmarks[i + 1])

    def reset_drawable_image(self):
        self.copied_image_for_drawing = self.image.copy()

    def draw_rectangle(self, x_y_w_h_tuple, color=(255, 0, 0)):
        (x, y, w, h) = x_y_w_h_tuple
        cv2.rectangle(self.copied_image_for_drawing, (x, y), (x + w, y + h), color, 2)

    def draw_cirle_by_coordinate(self, coordinates, color=(0, 255, 255), radius=1):
        cv2.circle(self.copied_image_for_drawing, coordinates, radius, color, -1)

    def draw_line(self, start_coordinates, end_coordinates, color=(0, 255, 0)):
        cv2.line(
            self.copied_image_for_drawing,
            tuple(start_coordinates),
            tuple(end_coordinates),
            color,
            2,
        )

    def put_text(self, text, coordinates=(20, 20), color=(0, 0, 0)):
        cv2.putText(
            self.copied_image_for_drawing,
            text,
            coordinates,
            cv2.FONT_HERSHEY_SIMPLEX,
            color=color,
            fontScale=0.5,
            thickness=1,
        )

    def display(self):
        fa.FaceAnalyzer().display_image(
            self.copied_image_for_drawing,
            (
                "Participant"
                if self.is_categorized_by_participant
                else " Frame" + str(self.index)
            ),
        )

    def resize(self, new_width: int, new_height: int = None):
        height, width = self.image.shape[:2]

        if new_height is None:
            # Only width is given -> preserve aspect ratio
            aspect_ratio = height / width
            new_height = int(new_width * aspect_ratio)

        self.image = cv2.resize(
            self.image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )

    def align_face_with_mediapipe_landmarks(self)-> None:
        if not self.facial_landmarks:
            return

        lm_forehead = self.facial_landmarks[10]
        lm_chin = self.facial_landmarks[152]
        x1, y1 = denormalize_landmarks_without_Z(lm_forehead, self.image)
        x2, y2 = denormalize_landmarks_without_Z(lm_chin, self.image)
        dx = x2 - x1
        dy = y2 - y1

        angle = -math.degrees(math.atan2(dx, dy))

        img_h, img_w = self.image.shape[:2]
        center = (img_w // 2, img_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.image = cv2.warpAffine(
            self.image, rotation_matrix, (img_w, img_h), flags=cv2.INTER_LINEAR
        )
        self._rotate_landmarks(rotation_matrix)

    def _rotate_landmarks(self, rotation_matrix):
        for landmark in self.facial_landmarks:
            x, y = denormalize_landmarks_without_Z(landmark, self.image)
            img_h, img_w = self.image.shape[:2]
            rotated_point = np.dot(rotation_matrix, np.array([x, y, 1]))

            landmark.x = rotated_point[0] / img_w
            landmark.y = rotated_point[1] / img_h

    def get_cropped_image(self, x1, y1, x2, y2):
        """
        Crop the image to the given coordinates (x1, y1, x2, y2) using OpenCV. 

        Args:
            x1, y1: The top-left corner.
            x2, y2: The bottom-right corner.

        Returns:
            Cropped image (NumPy array) or None if the coordinates are out of bounds.
        """
        img_height, img_width = self.image.shape[:2]
        if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
            return None
        return self.image[y1:y2, x1:x2]

    def is_blurry(self, threshold=50):  
        """
        takes greyscale image and threshold as input
        """
        img = self.image
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        print(f"Laplacian variance: {variance:.2f}")
        
        return variance < threshold

