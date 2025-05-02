import re
from typing import List, Tuple

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
import os

import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def get_labels_dict(participant_id: str):
    df = pd.read_csv(
        os.path.join(BASE_DIR, "data", "external", "turker_scores_full_interview.csv"),
    )
    df = df.loc[
        (df["Participant"] == participant_id.lower()) & (df["Worker"] == "AGGR")
    ]
    return df.iloc[0].to_dict()

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

def get_p_and_pp_participant_number():
    VIDEOS_FOLDER = os.path.join(BASE_DIR, "data", "raw", "videos")

    pp_pattern = re.compile(
        r"^PP(\d+)", re.IGNORECASE  # 'PP' at start, followed by digits
    )

    p_pattern = re.compile(
        r"(?<!P)P(\d+)", re.IGNORECASE  # 'P' not preceded by P/p, followed by digits
    )

    p_participant_numbers = []
    pp_participant_numbers = []

    # Loop over all files in the folder
    for filename in sorted(os.listdir(VIDEOS_FOLDER)):
        if(any(filename.endswith(ext) for ext in [".mp4", ".avi", ".mov"])):
            if pp_match := pp_pattern.search(filename):
                pp_participant_numbers.append(int(pp_match.group(1)))
            else:
                p_match = p_pattern.search(filename)
                p_participant_numbers.append(int(p_match.group(1)))

    # Sort the participant numbers
    p_participant_numbers.sort()
    pp_participant_numbers.sort()

    return p_participant_numbers, pp_participant_numbers

def get_participant_ids():
    p_participant_numbers, pp_participant_numbers = get_p_and_pp_participant_number()
    participant_ids = []
    for prefix, participant_numbers in [
        ("P", p_participant_numbers),
        ("PP", pp_participant_numbers),
    ]:
        for participant_number in participant_numbers:
            participant_id = f"{prefix}{participant_number}"
            participant_ids.append(participant_id)
    return participant_ids

def get_participant_dir(participant_id):
    output_dir = os.path.join(BASE_DIR, "data", "processed", "videos_frames")
    return os.path.join(output_dir, participant_id)