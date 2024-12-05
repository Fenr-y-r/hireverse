import matplotlib.pyplot as plt
import cv2
import numpy as np

class Frame:
    def __init__(self, index, participant, image):
        self.index = index            
        self.participant = participant 
        self.image = image
        self.smile_area = None

    def display_frame(self):
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {self.index}")
        plt.axis('off')
        plt.show()

    def __str__(self):
        return f"index={self.index}, smile_area={self.smile_area}, participant={self.participant}"