import os
import re
from typing import List, Tuple
import pandas as pd
from hireverse.utils.utils import BASE_DIR
import cv2
import numpy as np
from natsort import natsorted

class DatasetHandler:
    @staticmethod
    def get_labels_dict(participant_id: str):
        df = pd.read_csv(
            os.path.join(BASE_DIR, "data", "external", "turker_scores_full_interview.csv"),
        )
        df = df.loc[
            (df["Participant"] == participant_id.lower()) & (df["Worker"] == "AGGR")
        ]
        return df.iloc[0].to_dict()
    
    @staticmethod
    def get_participant_ids():
        p_participant_numbers, pp_participant_numbers = DatasetHandler._get_p_and_pp_participant_number()
        participant_ids = []
        for prefix, participant_numbers in [
            ("P", p_participant_numbers),
            ("PP", pp_participant_numbers),
        ]:
            for participant_number in participant_numbers:
                participant_id = f"{prefix}{participant_number}"
                participant_ids.append(participant_id)
        return participant_ids
    
    @staticmethod
    def _get_p_and_pp_participant_number() -> Tuple[List[int], List[int]]:
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
    
    def get_participant_dir(participant_id):
        output_dir = os.path.join(BASE_DIR, "data", "processed", "videos_frames")
        return os.path.join(output_dir, participant_id)
    
    @staticmethod
    def get_sorted_participant_frames_images(participant_id):
        participant_dir = DatasetHandler.get_participant_dir(participant_id)
        frames = []
        for filename in natsorted(os.listdir(participant_dir)):
            if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                image_path = os.path.join(participant_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    frames.append(image)
        return frames
    