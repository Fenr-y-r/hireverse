import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np
from schemas.selected_facial_landmarks import (
    TwoLandmarksConnector,
)
import os
from utils.utils import denormalize_landmarks_without_Z


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
        self.facial_landmarks = None
        self.copied_image_for_drawing = None
        self.two_landmarks_connectors: List[TwoLandmarksConnector]
        self.image = image
        self.faces = None
        self.facial_landmarks_obj = None
        self.face_angles: Tuple[int, int, int] = None
        self.is_categorized_by_participant = is_categorized_by_participant

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

    def draw_circle_by_facial_landmark(self, landmark):
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
                        self.draw_circle_by_facial_landmark(producing_landmarks[i])
                        self.draw_circle_by_facial_landmark(producing_landmarks[i + 1])

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
        import utils.face_analyzer as fa
        fa.FaceAnalyzer().display_image(
            self.copied_image_for_drawing,
            (
                "Participant"
                if self.is_categorized_by_participant
                else " Frame" + str(self.index)
            ),
        )
