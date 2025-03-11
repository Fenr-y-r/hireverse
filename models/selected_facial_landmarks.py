import math
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark


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
