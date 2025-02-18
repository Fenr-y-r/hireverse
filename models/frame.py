import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np

from models.selected_facial_landmarks import SelectedFacialLandmarks

class Frame:

    def __init__(self, index: int, participant: str, image: np.ndarray):
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
        self.faces= None
        
   

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
        print(self.face)
        if self.face is not None:
            self.draw_rectangle(self.face, (255, 0, 0))

    def draw_facial_landmarks(self):
        self._create_drawable_image_copy_if_not_exist()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        if self.facial_landmarks:
            mp_drawing.draw_landmarks(
            self.copied_image_for_drawing,
            self.facial_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,  # Draw the full mesh
            landmark_drawing_spec=None,  # No special styling for landmarks
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
                  
    def draw_selected_facial_landmarks(self):
        self._create_drawable_image_copy_if_not_exist()
        if self.selected_facial_landmarks:
            self.draw_line(self.selected_facial_landmarks.outer_lip_above, self.selected_facial_landmarks.outer_lip_below,  color = (0, 155, 255))
            self.draw_line(self.selected_facial_landmarks.inner_lip_above, self.selected_facial_landmarks.inner_lip_below,  color = (0, 255, 255))
            self.draw_line(self.selected_facial_landmarks.lip_corner_right, self.selected_facial_landmarks.lip_corner_left,  color = (0, 155, 255))            

            for attr, value in self.selected_facial_landmarks.__dict__.items():
                if isinstance(value, Tuple):
                    self.draw_cirle(value)

    def reset_drawable_image(self):
        self.copied_image_for_drawing = self.image.copy()

    def draw_rectangle(self, x_y_w_h_tuple, color = (255, 0, 0)):
        (x, y, w, h) = x_y_w_h_tuple
        cv2.rectangle(self.copied_image_for_drawing, (x, y), (x + w, y + h), color, 2)

    def draw_cirle(self, coordinates, color = (0, 255, 255)):
        cv2.circle(self.copied_image_for_drawing, coordinates, 2, color, -1)

    def draw_line(self, start_coordinates, end_coordinates, color = (0, 255, 0)):
        cv2.line(self.copied_image_for_drawing, tuple(start_coordinates), tuple(end_coordinates), color, 2)

    def display(self):
        self._create_drawable_image_copy_if_not_exist()
        plt.imshow(cv2.cvtColor(self.copied_image_for_drawing, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.index}")
        plt.axis('off')
        plt.show()






   