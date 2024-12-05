import numpy as np
from frame import Frame

import cv2


class SmileDetector:
    """
    The SmileDetector class is a utility class responsible for processing frames, detecting faces and smiles, 
    and extracting relevant data. 

    """
    def __get_smile_area(self, smile):
        (x, y, w, h) = smile
        return w * h  

    def normalize_smile_area(self, min_smile_area, max_smile_area , smile_area):
        normalized_smile_area = (smile_area - min_smile_area) / (max_smile_area - min_smile_area) * 100 
        return normalized_smile_area

    def calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def get_face(self, gray_frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return  face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    def __get_smile(self, roi_gray):
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        return smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=20, minSize=(25, 25))

        

    def get_frame_smile_area(self, frame: Frame):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        faces = self.get_face(gray)


        if len(faces) == 0:
            return 0

        largest_face = max(faces, key=self.calculate_face_area)
        (x, y, w, h) = largest_face
        roi_gray = gray[y:y+h, x:x+w]

        smiles = self.__get_smile(roi_gray)
        if len(smiles) > 0:
            return self.__get_smile_area(smiles[0])
        else:
            return 0
        