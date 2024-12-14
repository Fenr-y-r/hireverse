import matplotlib.pyplot as plt
import cv2
from typing import Tuple

class Frame:
    def __init__(self, index, participant, image):
        self.index = index            
        self.participant = participant 
        self.image = image
        self.smile_area = None
        self.face = None
        self.smile = None
        self.facial_landmarks = None
        self.copied_image_for_drawing = None

    def __str__(self):
        return f"index={self.index}, smile_area={self.smile_area}, participant={self.participant}"

    def create_drawable_image_copy(self):
        self.copied_image_for_drawing = self.image.copy()

    def draw_smile(self):
        if self.smile is not None:
            self.draw_rectangle(self.smile, (0, 0, 255))

    def draw_face(self):
        if self.face is not None:
            self.draw_rectangle(self.face, (255, 0, 0))

    def draw_facial_landmarks(self):
        if self.facial_landmarks is not None:
            for (x, y) in self.facial_landmarks:
                self.draw_cirle((x,y))
    
    def draw_rectangle(self, x_y_w_h_tuple, color):
        (x, y, w, h) = x_y_w_h_tuple
        cv2.rectangle(self.copied_image_for_drawing, (x, y), (x + w, y + h), color, 2)

    def draw_cirle(self, coordinates: Tuple[int, int]):
        cv2.circle(self.copied_image_for_drawing, coordinates, 2, (0, 255, 0), -1)

    def display(self):
        plt.imshow(cv2.cvtColor(self.copied_image_for_drawing, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.index}")
        plt.axis('off')
        plt.show()