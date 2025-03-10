from dataclasses import dataclass, field
import math
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from typing import Optional, Tuple

from models.helper_facial_landmarks import HelperFacialLandmarks


@dataclass
class SelectedFacialLandmarks:
    helper_facial_landmarks: HelperFacialLandmarks
    outer_lip_height: float
    inner_lip_height: float
    lip_corner_distance: float
    average_outer_brow_height: float
    average_inner_brow_height: float
    eye_open: float


class TwoLandmarksConnector:
    def __init__(self, name, producing_landmarks):
        self.name: str = name
        self.producing_landmarks: list[NormalizedLandmark] = producing_landmarks
        self.length = self.get_length()

    def get_length(self):
        if len(self.producing_landmarks) == 2:
            return self._euclidean_distance_between_two_interest_points(
                self.producing_landmarks[0], self.producing_landmarks[1]
            )
        elif len(self.producing_landmarks) == 4:
            return (
                self._euclidean_distance_between_two_interest_points(
                    self.producing_landmarks[0], self.producing_landmarks[1]
                )
                + self._euclidean_distance_between_two_interest_points(
                    self.producing_landmarks[2], self.producing_landmarks[3]
                )
            ) / 2

    def _euclidean_distance_between_two_interest_points(self, point1, point2):
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)
