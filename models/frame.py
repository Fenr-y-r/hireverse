import matplotlib.pyplot as plt
import cv2
from typing import Tuple
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
        self.face_interest_points = None
        self.copied_image_for_drawing = None
        self.selected_facial_landmarks: SelectedFacialLandmarks = None

    def create_drawable_image_copy(self):
        self.copied_image_for_drawing = self.image.copy()

    def draw_smile(self):
        if self.smile:
            self.draw_rectangle(self.smile, (0, 255, 0))

    def draw_face(self):
        if self.face is not None:
            self.draw_rectangle(self.face, (255, 0, 0))

    def draw_facial_landmarks(self):
        if self.face_interest_points:
            for (x, y) in self.face_interest_points:
                self.draw_cirle((x,y))
    
    def draw_selected_facial_landmarks(self):
        if self.selected_facial_landmarks:
            self.draw_line(self.selected_facial_landmarks.outer_lip_above, self.selected_facial_landmarks.outer_lip_below,  color = (0, 155, 255))
            self.draw_line(self.selected_facial_landmarks.inner_lip_above, self.selected_facial_landmarks.inner_lip_below,  color = (0, 255, 255))
            self.draw_line(self.selected_facial_landmarks.lip_corner_right, self.selected_facial_landmarks.lip_corner_left,  color = (0, 155, 255))            

            for attr, value in self.selected_facial_landmarks.__dict__.items():
                if isinstance(value, Tuple):
                    self.draw_cirle(value)


    def draw_rectangle(self, x_y_w_h_tuple, color = (255, 0, 0)):
        (x, y, w, h) = x_y_w_h_tuple
        cv2.rectangle(self.copied_image_for_drawing, (x, y), (x + w, y + h), color, 2)

    def draw_cirle(self, coordinates, color = (0, 255, 255)):
        cv2.circle(self.copied_image_for_drawing, coordinates, 2, color, -1)

    def draw_line(self, start_coordinates, end_coordinates, color = (0, 255, 0)):
        cv2.line(self.copied_image_for_drawing, tuple(start_coordinates), tuple(end_coordinates), color, 2)

    def display(self):
        if self.copied_image_for_drawing is None:
            self.create_drawable_image_copy()
        plt.imshow(cv2.cvtColor(self.copied_image_for_drawing, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.index}")
        plt.axis('off')
        plt.show()

    