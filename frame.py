import matplotlib.pyplot as plt
import cv2
import numpy as np

class Frame:
    def __init__(self, index, participant, image):
        self.index = index            
        self.participant = participant 
        self.image = image
        self.smile_area = None
        self.face = None
        self.smile = None

    def display_frame(self):
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.index}")
        plt.axis('off')
        plt.show()

    def __str__(self):
        return f"index={self.index}, smile_area={self.smile_area}, participant={self.participant}"

    def draw(self):
        if self.smile is not None:
            # Draw a rectangle around the smile
            (x, y, w, h) = self.smile  # Extract coordinates of the smile
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

        # Optionally, draw the face rectangle if face is not None
        if self.face is not None:
            (fx, fy, fw, fh) = self.face  # Extract face coordinates
            cv2.rectangle(self.image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Draw blue rectangle