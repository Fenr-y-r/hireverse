import math
from typing import List, Tuple
import cv2
import dlib  # type: ignore
from imutils import face_utils  # type: ignore
import numpy as np
from models.selected_facial_landmarks import SelectedFacialLandmarks
from models.frame import Frame
from scipy.spatial.transform import Rotation as R
import mediapipe as mp


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

    def get_face_interest_points(self, frame: Frame):
        if frame.face is None:
            return None
        (x, y, w, h) = frame.face
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # uses a facial landmark predictor to detect landmarks within a specified region of the image
        landmarks = predictor(self.convert_to_gray(frame.image), dlib_rect)
        landmarks_np = face_utils.shape_to_np(landmarks)  # convert to a NumPy array
        return [
            (point[0], point[1]) for point in landmarks_np
        ]  # convert to a list of tuples

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

    # def get_head_pose(self, frame: Frame) -> Tuple[float, float, float]:
    #     """ Computes yaw, pitch, and roll (in degrees) for a given frame. """

    #     # Define 3D model points for a generic face (in millimeters)
    #     # later used in pose estimation, where the relationship between these 3D points and their 2D projections can be used to estimate the orientation and position of the face in 3D space relative to the camera.
    #     self.model_points = np.array([
    #         (0.0, 0.0, 0.0),         # Nose tip
    #         (-225.0, 170.0, -135.0), # Left eye left corner
    #         (225.0, 170.0, -135.0),  # Right eye right corner
    #         (-150.0, -150.0, -125.0),# Left mouth corner
    #         (150.0, -150.0, -125.0), # Right mouth corner
    #         (0.0, -330.0, -65.0)    # Chin
    #     ], dtype=np.float32)

    #     if frame.face is None or frame.facial_landmarks is None:
    #         return (0.0, 0.0, 0.0)

    #     (_, _, w, h) = frame.face
    #     landmarks_np = frame.facial_landmarks

    #     # Select 2D image points corresponding to 3D model points
    #     detected_points = np.array([
    #         landmarks_np[30],  # Nose tip
    #         landmarks_np[36],  # Left eye left corner
    #         landmarks_np[45],  # Right eye right corner
    #         landmarks_np[48],  # Left mouth corner
    #         landmarks_np[54],  # Right mouth corner
    #         landmarks_np[8]    # Chin
    #     ], dtype=np.float32)

    #     # Camera matrix (assuming focal length ~ image width, center is at (w/2, h/2))
    #     focal_length = w

    #     # Has information about the camera’s internal parameters, such as focal length and optical center
    #     # It is used to convert 3D world coordinates into 2D image coordinates based on the camera’s perspective.
    #     camera_matrix = np.array([
    #         [focal_length, 0, w / 2],
    #         [0, focal_length, h / 2],
    #         [0, 0, 1]
    #     ], dtype=np.float32)

    #     # These are the lens distortion coefficients. Cameras, especially with wide-angle lenses, introduce distortions like radial or tangential distortion. These coefficients are used to correct for such lens distortions in the image.
    #     dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    #     # Solve PnP to get rotation vector
    #     success, rvec, _ = cv2.solvePnP(
    #         self.model_points, detected_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    #     )

    #     if not success:
    #         return (0.0, 0.0, 0.0)

    #     # Convert rotation vector to rotation matrix
    #     rotation_matrix, _ = cv2.Rodrigues(rvec)

    #     # Convert rotation matrix to Euler angles (yaw, pitch, roll)
    #     rotation = R.from_matrix(rotation_matrix)
    #     yaw, pitch, roll = rotation.as_euler('xyz', degrees=True)

    #     return yaw, pitch, roll

    # TODO: change isWebcam
    def get_head_pose_using_mediapipe(self, image, isWebcam=False):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, min_detection_confidence=0.5
        )

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        if isWebcam:
            image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Performance improvement when passed to model
        image.flags.writeable = False

        results = face_mesh.process(image)  # pass to model

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to draw on it using opencv

        img_h, img_w, img_channels_number = image.shape
        face_3d = []
        face_2d = []

        # TODO: only work on largest face
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i, landmark in enumerate(face_landmarks.landmark):
                    if i == 33 or i == 263 or i == 1 or i == 61 or i == 291 or i == 199:
                        if i == 1:
                            nose_2d = (landmark.x * img_w, landmark.y * img_h)
                            nose_3d = (landmark.x, landmark.y, landmark.z * 3000)
                        x, y = int(landmark.x * img_w), int(
                            landmark.y * img_h
                        )  # x and y values are normalized, so we need to convert htem back to image coordinates by scaling them
                        face_2d.append([x, y])
                        #  z is the 3D depth of the landmark, and it's inferred based on face geometry and landmark pos
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

                x= angles[0] * 360 # because normalized
                y= angles[1] * 360
                z= angles[2] * 360

                if y < -10:
                    text= "Looking left"
                elif y > 10:
                    text= "Looking down"
                elif x>10:
                    text= "Looking down"
                elif x<-10:
                    text= "Looking up"
                else:
                    text= "Looking straight"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, distortion_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))

                cv2.line(image,p1, p2, (255,0,0), 3)

                

                cv2.putText(image, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, "X: " + str(x), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, "Y: " + str(y), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, "Z: " + str(z), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec
                )
                cv2.imshow('Head Pose', image)  
                cv2.waitKey(0)  # Waits indefinitely for a key press
                cv2.destroyAllWindows()  


