import math
from typing import List, Tuple
import cv2
import dlib  # type: ignore
from imutils import face_utils  # type: ignore

from models.selected_facial_landmarks import SelectedFacialLandmarks
from models.frame import Frame


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
        return tuple(max(faces, key=self.calculate_face_area)) if len(faces) else None

    """This returns the coordinates of one smile relative to the face not the whole frame"""

    def get_smile(self, frame_image, face):
        (x, y, w, h) = face
        gray_image = self.convert_to_gray(frame_image)
        roi_gray = gray_image[y : y + h, x : x + w]
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=35, 
        )
        if len(smiles):
            largest_smile = max(smiles, key=lambda s: s[2] * s[3])
            (sx, sy, sw, sh) = largest_smile
            if sy > h // 2:
                return (x + sx, y + sy, sw, sh)
        
        return None

    def get_face_interest_points(self, frame: Frame):
        if frame.face is None:
            return None
        (x, y, w, h) = frame.face
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # uses a facial landmark predictor to detect landmarks within a specified region of the image
        landmarks = predictor(self.convert_to_gray(frame.image), dlib_rect)
        landmarks_np = face_utils.shape_to_np(landmarks)  # convert to a NumPy array
        return [(point[0], point[1]) for point in landmarks_np]  # convert to a list of tuples

    def _get_brow_interest_points(
        self, face_interest_points: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        outer_brow_left = face_interest_points[17]
        inner_brow_left = face_interest_points[21]
        inner_brow_right = face_interest_points[22]
        outer_brow_right = face_interest_points[26]
        return (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right)

    def _get_eye_interest_points(
        self, face_interest_points: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        eye_outer_left = face_interest_points[36]
        eye_outer_right = face_interest_points[45]
        eye_inner_left = face_interest_points[39]
        eye_inner_right = face_interest_points[42]
        return (eye_outer_left, eye_outer_right, eye_inner_left, eye_inner_right)

    def get_lips_coordinates(self, face_interest_points: List[Tuple[int, int]]):
        outer_lip_above = face_interest_points[51]
        outer_lip_below = face_interest_points[57]
        inner_lip_above = face_interest_points[62]
        inner_lip_below = face_interest_points[66]
        lip_corner_right = face_interest_points[54]
        lip_corner_left = face_interest_points[48]
        return (
            outer_lip_above,
            outer_lip_below,
            inner_lip_above,
            inner_lip_below,
            lip_corner_right,
            lip_corner_left,
        )

    def _euclidean_distance_between_two_interest_points(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def _get_lip_lengths(self, face_interest_points: List[Tuple[int, int]]):
        (
            outer_lip_above,
            outer_lip_below,
            inner_lip_above,
            inner_lip_below,
            lip_corner_right,
            lip_corner_left,
        ) = self.get_lips_coordinates(face_interest_points)
        outer_lip_height = self._euclidean_distance_between_two_interest_points(
            outer_lip_above, outer_lip_below
        )
        inner_lip_height = self._euclidean_distance_between_two_interest_points(
            inner_lip_above, inner_lip_below
        )
        lip_corner_distance = self._euclidean_distance_between_two_interest_points(
            lip_corner_left, lip_corner_right
        )
        return (outer_lip_height, inner_lip_height, lip_corner_distance)

    def get_selected_facial_features(
        self, face_interest_points: List[Tuple[int, int]]
    ) -> SelectedFacialLandmarks:
        (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right) = (
            self._get_brow_interest_points(face_interest_points)
        )
        (eye_outer_left, eye_outer_right, eye_inner_left, eye_inner_right) = (
            self._get_eye_interest_points(face_interest_points)
        )
        (outer_lip_height, inner_lip_height, lip_corner_distance) = (
            self._get_lip_lengths(face_interest_points)
        )
        (
            outer_lip_above,
            outer_lip_below,
            inner_lip_above,
            inner_lip_below,
            lip_corner_right,
            lip_corner_left,
        ) = self.get_lips_coordinates(face_interest_points)
        return SelectedFacialLandmarks(
            inner_brow_left=inner_brow_left,
            outer_brow_left=outer_brow_left,
            inner_brow_right=inner_brow_right,
            outer_brow_right=outer_brow_right,
            eye_outer_left=eye_outer_left,
            eye_outer_right=eye_outer_right,
            eye_inner_left=eye_inner_left,
            eye_inner_right=eye_inner_right,
            outer_lip_height=outer_lip_height,
            inner_lip_height=inner_lip_height,
            lip_corner_distance=lip_corner_distance,
            outer_lip_above=outer_lip_above,
            outer_lip_below=outer_lip_below,
            inner_lip_above=inner_lip_above,
            inner_lip_below=inner_lip_below,
            lip_corner_right=lip_corner_right,
            lip_corner_left=lip_corner_left,
        )
