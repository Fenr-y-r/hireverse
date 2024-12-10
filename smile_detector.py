import numpy as np
from frame import Frame

import cv2


class SmileDetector:
    """
    The SmileDetector class is a utility class responsible for processing frames, detecting faces and smiles,
    and extracting relevant data.

    """

    def get_smile_area(self, smile):
        (x, y, w, h) = smile
        return w * h

    def normalize_smile_area(self, min_smile_area, max_smile_area, smile_area):
        if(max_smile_area == min_smile_area):
            return smile_area
        normalized_smile_area = (
            (smile_area - min_smile_area) / (max_smile_area - min_smile_area) * 100
        )
        return normalized_smile_area

    def calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def get_face(self, frame_image):
        gray_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return max(faces, key=self.calculate_face_area) if len(faces) else None

    """This returns the coordinates of one smile relative to the face not the whole frame"""
    def get_smile(self, frame_image, largest_face):
        (x, y, w, h) = largest_face
        gray_frame = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray_frame[y : y + h, x : x + w]
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=20, minSize=(25, 25)
        )
        if len(smiles):
            index = 0
            if len(smiles) > 1:
                index = 1
            (sx, sy, sw, sh) = smiles[index]    
            return (x + sx, y + sy, sw, sh)
        else:
            return None
    
   
    
