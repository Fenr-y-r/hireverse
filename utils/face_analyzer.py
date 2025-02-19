import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

from models.frame import Frame
from models.selected_facial_landmarks import SelectedFacialLandmarks
from utils.utils import denormalize_int

class FaceAnalyzer:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, min_detection_confidence=0.5
    )        


    def _calculate_face_area(self, face):
        (x, y, w, h) = face
        return w * h

    def get_smile(self, frame_image, face):
        (x, y, w, h) = face
        gray_image = self.convert_to_gray(frame_image)
        roi_gray = gray_image[y : y + h, x : x + w]
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.04,  # Compensates for that an object at different distances from the camera will appear at different sizes. A lower scaleFactor increases the detection time but also increases the chance of detection.  # Typical values range from 1.01 to 1.3.
            minNeighbors=47,  # Higher values result in fewer detections but with higher quality. Lower values may lead to more detections but with possible false positives. It’s a trade-off between precision and recall.
            # minSize=(30, 30)    # smiles smaller than this size are ignored.  # TODO: change according to the face distance from webcam
        )
        if len(smiles):
            largest_smile = max(smiles, key=lambda s: s[2] * s[3])
            (sx, sy, sw, sh) = largest_smile
            if sy > h // 2:
                return (x + sx, y + sy, sw, sh)

        return None

    def get_face_coordinates(
        self, face_landmarks: list[NormalizedLandmark] , image
    ):
        """
        This method returns the coordinates of the face in the form of a tuple (x, y, w, h).
        """
    
        # Calculate the bounding box for each face
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]
        # TODO: replace the follwoing with   bboxC = detection.location_data.relative_bounding_box
        # Calculate the bounding box coordinates (normalized)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        img_h = image.shape[1]
        img_w = image.shape[0]
        return (
            int(min_x * img_h),
            int(min_y * img_w),
            round((max_x - min_x) * img_h),
            round((max_y - min_y) * img_w),
        )


    def process_image_results (self,image) :
        image.flags.writeable = False
        results = FaceAnalyzer.face_mesh.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).multi_face_landmarks
        image.flags.writeable = True
        return results

    # TODO: make sure this works
    def get_largest_face_landmarks_obj(self, image, detected_faces_landmarks) -> list[NormalizedLandmark]:
        """
        This function takes the MediaPipe results and returns the largest face landmarks
        based on bounding box area.
        """

        if detected_faces_landmarks:
            max_area = 0
            largest_face_landmarks = None

            for face_landmarks_obj in detected_faces_landmarks:
                _, _, w, h = self.get_face_coordinates(face_landmarks_obj.landmark, image)
                area = w * h

                if area > max_area:
                    max_area = area
                    largest_face_landmarks = face_landmarks_obj
            return largest_face_landmarks

        return None

    def _get_brow_interest_points(
        self, face_interest_points
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        outer_brow_left = face_interest_points[276]
        inner_brow_left = face_interest_points[285]

        inner_brow_right = face_interest_points[55]
        outer_brow_right = face_interest_points[46]
        return (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right)

    def _get_eye_interest_points(
        self, face_interest_points: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        eye_outer_left = face_interest_points[33]
        eye_inner_left = face_interest_points[133]
        eye_inner_right = face_interest_points[362]
        eye_outer_right = face_interest_points[263]
        return (eye_outer_left, eye_outer_right, eye_inner_left, eye_inner_right)

    def get_lips_coordinates(self, face_interest_points: List[Tuple[int, int]]):
        outer_upper_lip = face_interest_points[0]
        inner_upper_lip = face_interest_points[13]
        inner_lower_lip = face_interest_points[14]
        outer_lower_lip = face_interest_points[17]
        lip_corner_right = face_interest_points[291]
        lip_corner_left = face_interest_points[61]
        return (
            outer_upper_lip,
            outer_lower_lip,
            inner_upper_lip,
            inner_lower_lip,
            lip_corner_right,
            lip_corner_left,
        )

    def _euclidean_distance_between_two_interest_points(self, point1, point2):
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

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

    def get_selected_facial_landmarks(
        self, face_interest_points: List[Tuple[int, int, int]]
    ) -> SelectedFacialLandmarks:
        if face_interest_points:
            (outer_brow_left, inner_brow_left, inner_brow_right, outer_brow_right) = (self._get_brow_interest_points(face_interest_points))
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
        return None
    
    def get_face_angles(self, image, face_landmarks: list[NormalizedLandmark], isWebcam=False):
        """
        returns a rounded tuple (x, y, z) representing the angles of the face in degrees.
        """
        # if isWebcam:
        #     image = cv2.flip(image, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if face_landmarks is None:
            return None
        img_h, img_w, img_channels_number = image.shape
        face_3d = []
        face_2d = []

        for i, landmark in enumerate(face_landmarks):
            if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:
                # if i == 1:
                    # nose_2d = (landmark.x * img_w, landmark.y * img_h)
                    # nose_3d = (landmark.x, landmark.y, landmark.z * 3000)
                x, y = denormalize_int(landmark.x, img_w), denormalize_int(landmark.y, img_h)
                face_2d.append([x, y])
                # Z is an estimated depth coordinate relative to the camera calculated using a machine learning mode. It is not an absolute distance, but rather a value indicating how far a landmark is from the image plane.
                # The model does not compute actual physical depth (in meters or cm). Instead, it predicts a relative depth value, where larger negative Z-values mean landmarks are deeper (farther from the camera).
                # The Z-values are normalized relative to the bounding box size of the detected face. If the face appears closer to the camera, the bounding box is larger, and the Z-values may be scaled differently.
                face_3d.append([x, y, landmark.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array(
            [
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1],
            ],
            dtype="double",
        )
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)  # no distortion
        #  find out where the camera is in relation to the object and its orientation.
        # the output rotation and translation vectors tell you how the object is oriented and positioned relative to the camera in the 3D world.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, distortion_matrix
        )
        rotation_matrix, jacobian_matrix = cv2.Rodrigues(rotation_vector)
        angles, mtx, dist, rvecs, tvecs, _ = cv2.RQDecomp3x3(rotation_matrix)

        x = angles[0] * 360 # because normalized
        y = angles[1] * 360
        z = angles[2] * 360

        if y < -10:
            text = "Looking left"
        elif y > 10:
            text = "Looking down"
        elif x > 10:
            text = "Looking down"
        elif x < -10:
            text = "Looking up"
        else:
            text = "Looking straight"

        # nose_3d_projection, jacobian = cv2.projectPoints(
        #     nose_3d,
        #     rotation_vector,
        #     translation_vector,
        #     cam_matrix,
        #     distortion_matrix,
        # )

        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
        # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

        # cv2.line(image, p1, p2, (255, 0, 0), 3)

        # cv2.putText(
        #     image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        # )
        # cv2.putText(
        #     image,
        #     "X: " + str(x),
        #     (20, 100),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 0),
        #     2,
        # )
        # cv2.putText(
        #     image,
        #     "Y: " + str(y),
        #     (20, 150),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 0),
        #     2,
        # )
        # cv2.putText(
        #     image,
        #     "Z: " + str(z),
        #     (20, 200),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 0),
        #     2,
        # )

        return (x, y, z)