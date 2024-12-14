
import cv2
import dlib  # type: ignore
from imutils import face_utils  # type: ignore

from eye_brows import EyeBrows
from facial_features import FacialFeatures
from frame import Frame


class FaceAnalyzer:

    def get_smile_area(self, smile):
        (x, y, w, h) = smile
        return w * h

    def calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def convert_to_gray(self, frame_image):
        return cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

    def get_face(self, frame_image):
        gray_image = self.convert_to_gray(frame_image)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
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

    def get_face_interest_points(self, frame: Frame):
        if frame.face is None:
            return None
        (x, y, w, h) = frame.face
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # uses a facial landmark predictor to detect landmarks within a specified region of the image
        landmarks = predictor(self.convert_to_gray(frame.image), dlib_rect)
        return face_utils.shape_to_np(landmarks)  # convert to a NumPy array
    
    def get_eyebrows_coordinates(self, landmarks):
        outer_brow_left = landmarks[17]
        inner_brow_left = landmarks[21]
        inner_brow_right = landmarks[22]
        outer_brow_right = landmarks[26]
        return EyeBrows(
            inner_brow_left, outer_brow_left, inner_brow_right, outer_brow_right
        )
