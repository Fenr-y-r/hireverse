import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np
from utils.utils import denormalize_landmarks_without_Z
from models.selected_facial_landmarks import SelectedFacialLandmarks


class Frame:

    def __init__(self, index: int, participant: str, image: np.ndarray, is_categorized_by_participant: bool = False):
        self.index = index
        self.participant = participant
        self.image = image
        self.smile_area = None
        self.face: Tuple[int, int, int, int] = None
        self.smile: Tuple[int, int, int, int] = None
        self.facial_landmarks = None
        self.copied_image_for_drawing = None
        self.selected_facial_landmarks: SelectedFacialLandmarks = None
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

    def draw_face(self):
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
            for i, angle_name in enumerate("XY"):
                self.put_text(f"{angle_name}: {round(self.face_angles[i], 1)}", (20, 20 + i * 20))

    def draw_selected_facial_landmarks(self, draw_lines=True):
        self._create_drawable_image_copy_if_not_exist()
        if self.selected_facial_landmarks:
            if draw_lines:
                self.draw_line(
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.outer_lip_above, self.image
                    ),
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.outer_lip_below, self.image
                    ),
                    color=(0, 155, 255),
                )
                self.draw_line(
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.inner_lip_above, self.image
                    ),
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.inner_lip_below, self.image
                    ),
                    color=(0, 255, 255),
                )
                self.draw_line(
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.lip_corner_right, self.image
                    ),
                    denormalize_landmarks_without_Z(
                        self.selected_facial_landmarks.lip_corner_left, self.image
                    ),
                    color=(0, 155, 255),
                )

            for attr, value in self.selected_facial_landmarks.__dict__.items():
                if type(value).__name__ == "NormalizedLandmark":
                    self.draw_cirle(denormalize_landmarks_without_Z(value, self.image))

    def reset_drawable_image(self):
        self.copied_image_for_drawing = self.image.copy()

    def draw_rectangle(self, x_y_w_h_tuple, color=(255, 0, 0)):
        (x, y, w, h) = x_y_w_h_tuple
        cv2.rectangle(self.copied_image_for_drawing, (x, y), (x + w, y + h), color, 2)

    def draw_cirle(self, coordinates, color=(0, 255, 255), radius=1):
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
        self._create_drawable_image_copy_if_not_exist()
        plt.imshow(cv2.cvtColor(self.copied_image_for_drawing, cv2.COLOR_BGR2RGB))
        plt.title(("Participant" if self.is_categorized_by_participant else " Frame") +  str(self.index))
        plt.axis("off")
        plt.show()
