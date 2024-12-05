import cv2
import numpy as np

from frame import Frame


class SmileDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def __get_smile_area(self, smile):
        (x, y, w, h) = smile
        return w * h  

    def normalize_smile_intensities(self, smile_areas):
        min_smile_area = min(smile_areas)
        max_smile_area = max(smile_areas)
        normalized_smile_intensities = [(area - min_smile_area) / (max_smile_area - min_smile_area) * 100 for area in smile_areas]
        return normalized_smile_intensities

    def calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def __detect_faces(self, gray_frame):
        return self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def __detect_smiles(self, roi_gray):
        return self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=20, minSize=(25, 25))

    def set_frame_smile_area(self, frame: Frame):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        faces = self.__detect_faces(gray)

        if len(faces) == 0:
            return 0

        largest_face = max(faces, key=self.calculate_face_area)
        (x, y, w, h) = largest_face
        roi_gray = gray[y:y+h, x:x+w]

        smiles = self.__detect_smiles(roi_gray)
        if len(smiles) > 0:
            frame.smile_area = self.__get_smile_area(smiles[0])
        else:
            frame.smile_area = 0
        
    def set_average_smile_intensity(self, smile_areas):
        self.average_smile_intensity = np.mean(smile_areas)