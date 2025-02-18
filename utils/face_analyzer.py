import math
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from typing import List, Tuple
import numpy as np

from models.frame import Frame
from models.selected_facial_landmarks import SelectedFacialLandmarks

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
        self, face_landmarks, image
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

    def process_image (self,image):
        return FaceAnalyzer.face_mesh.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ).multi_face_landmarks

    # TODO: make sure this works
    def get_largest_face_landmarks(self, image, detected_faces_landmarks):
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

    def get_facial_landmarks(self, image, detected_faces_landmarks):
        largest_face_landmarks = self.get_largest_face_landmarks(image, detected_faces_landmarks)
        return largest_face_landmarks
    










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

    def get_selected_facial_landmarks(
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

 
    
    
    # def get_head_pose_using_mediapipe(self, image, isWebcam=False):

    #     mp_drawing = mp.solutions.drawing_utils
    #     drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    #     if isWebcam:
    #         image = cv2.flip(image, 1)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Performance improvement when passed to model
    #     image.flags.writeable = False

    #     results = FaceAnalyzer.face_mesh.process(image)  # pass to model

    #     image.flags.writeable = False
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to draw on it using opencv

    #     img_h, img_w, img_channels_number = image.shape
    #     face_3d = []
    #     face_2d = []

    #     # TODO: only work on largest face
    #     if results.multi_face_landmarks:
    #         for face_landmarks in results.multi_face_landmarks:
    #             for i, landmark in enumerate(face_landmarks.landmark):
    #                 if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:
    #                     if i == 1:
    #                         nose_2d = (landmark.x * img_w, landmark.y * img_h)
    #                         nose_3d = (landmark.x, landmark.y, landmark.z * 3000)
    #                     x, y = int(landmark.x * img_w), int(
    #                         landmark.y * img_h
    #                     )  # x and y values are normalized, so we need to convert htem back to image coordinates by scaling them
    #                     face_2d.append([x, y])
    #                     #  z is the 3D depth of the landmark, and it's inferred based on face geometry and landmark pos
    #                     face_3d.append([x, y, landmark.z])

    #             face_2d = np.array(face_2d, dtype=np.float64)
    #             face_3d = np.array(face_3d, dtype=np.float64)

    #             focal_length = img_w
    #             cam_matrix = np.array(
    #                 [
    #                     [focal_length, 0, img_h / 2],
    #                     [0, focal_length, img_w / 2],
    #                     [0, 0, 1],
    #                 ],
    #                 dtype="double",
    #             )
    #             distortion_matrix = np.zeros((4, 1), dtype=np.float64)  # no distortion
    #             #  find out where the camera is in relation to the object and its orientation.
    #             # the output rotation and translation vectors tell you how the object is oriented and positioned relative to the camera in the 3D world.
    #             success, rotation_vector, translation_vector = cv2.solvePnP(
    #                 face_3d, face_2d, cam_matrix, distortion_matrix
    #             )
    #             rotation_matrix, jacobian_matrix = cv2.Rodrigues(rotation_vector)
    #             angles, mtx, dist, rvecs, tvecs, _ = cv2.RQDecomp3x3(rotation_matrix)

    #             x = angles[0] * 360  # because normalized
    #             y = angles[1] * 360
    #             z = angles[2] * 360

    #             if y < -10:
    #                 text = "Looking left"
    #             elif y > 10:
    #                 text = "Looking down"
    #             elif x > 10:
    #                 text = "Looking down"
    #             elif x < -10:
    #                 text = "Looking up"
    #             else:
    #                 text = "Looking straight"

    #             nose_3d_projection, jacobian = cv2.projectPoints(
    #                 nose_3d,
    #                 rotation_vector,
    #                 translation_vector,
    #                 cam_matrix,
    #                 distortion_matrix,
    #             )

    #             p1 = (int(nose_2d[0]), int(nose_2d[1]))
    #             p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    #             cv2.line(image, p1, p2, (255, 0, 0), 3)

    #             cv2.putText(
    #                 image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    #             )
    #             cv2.putText(
    #                 image,
    #                 "X: " + str(x),
    #                 (20, 100),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1,
    #                 (0, 0, 0),
    #                 2,
    #             )
    #             cv2.putText(
    #                 image,
    #                 "Y: " + str(y),
    #                 (20, 150),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1,
    #                 (0, 0, 0),
    #                 2,
    #             )
    #             cv2.putText(
    #                 image,
    #                 "Z: " + str(z),
    #                 (20, 200),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 1,
    #                 (0, 0, 0),
    #                 2,
    #             )

    #             mp_drawing.draw_landmarks(
    #                 image,
    #                 face_landmarks,
    #                 mp_face_mesh.FACEMESH_TESSELATION,
    #                 drawing_spec,
    #             )
    #             cv2.imshow("Head Pose", image)
    #             cv2.waitKey(0)  # Waits indefinitely for a key press
    #             cv2.destroyAllWindows()
