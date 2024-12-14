
import cv2
import dlib
from imutils import face_utils

from frame import Frame

# ### Cascade classifiers

# The “cascade” in cascade classifiers refers to the process where the input image is passed through a series of stages or filters.
# Each stage uses a weak classifier to determine if the object of interest (e.g., a face or a smile) is present.
# If the image passes through a stage, it proceeds to the next stage. If it fails a stage, the image is rejected.

# - Haar Features: The classifier uses Haar-like features to detect objects.
#   They are a set of simple rectangular features used in computer vision, particularly for object detection, such as face detection.

# ### Grayscale

# - Converting an image to grayscale is a common preprocessing step in computer vision tasks. Reasons:

#   1. Simplification: Grayscale has only one channel compared to the three channels (RGB). This makesprocessing faster and more efficient.

#   2. Feature Extraction:computer vision algorithms work better on grayscale images because they rely on intensity changes rather than color changes.

#   3. Noise Reduction


class SmileDetector:
    """
    The SmileDetector class is a utility class responsible for processing frames, detecting faces and smiles,
    and extracting relevant data.

    """

    def get_smile_area(self, smile):
        (x, y, w, h) = smile
        return w * h

    def normalize_smile_area(self, min_smile_area, max_smile_area, smile_area):
        if max_smile_area == min_smile_area:
            return smile_area
        normalized_smile_area = (
            (smile_area - min_smile_area) / (max_smile_area - min_smile_area) * 100
        )
        return normalized_smile_area

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

    def get_face_landmarks(self, frame: Frame):
        if frame.face is None:
            return None
        (x, y, w, h) = frame.face
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # uses a facial landmark predictor to detect landmarks within a specified region of the image
        landmarks = predictor(self.convert_to_gray(frame.image), dlib_rect)
        return face_utils.shape_to_np(landmarks)  # convert to a NumPy array