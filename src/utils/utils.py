from typing import List, Tuple

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def denormalize_landmarks_without_Z(
    landmark: NormalizedLandmark, img
) -> Tuple[int, int]:
    """
    Denormalize the landmarks to the original image size and ignores Z axis.
    """
    img_w, img_h = img.shape[1], img.shape[0]
    ordered_pair = landmark
    return (
        denormalize_int(ordered_pair.x, img_w),
        denormalize_int(ordered_pair.y, img_h),
    )


def denormalize_int(normalized_value, normalization_factor):
    """
    Denormalize the value to the original size and return it as an integer.
    """
    return int(normalized_value * normalization_factor)
