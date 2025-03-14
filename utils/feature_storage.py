from dataclasses import asdict
import numpy as np
import pandas as pd
import os

from models.frame import Frame
from models.model_features import *


class FeatureStorage:

    def __init__(self, csv_path: str ):
        if csv_path:
            self.csv_path = csv_path

    @classmethod
    def save_to_csv(cls, participant_id: str, **features):
        data = {"participant_id": participant_id}

        for feature in features.values():
            data.update(asdict(feature))

        df = pd.DataFrame([data])

        if not os.path.exists(cls.csv_path):
            # Create new file if it doesn’t exist
            df.to_csv(cls.csv_path, index=False)
        else:
            existing_df = pd.read_csv(cls.csv_path)
            existing_ids = set(
                existing_df["participant_id"]
            )  # # Convert participant_id column to a set for O(1) lookup
            if participant_id in existing_ids:
                return
            df.to_csv(
                cls.csv_path, mode="a", header=False, index=False
            )  # Append new row

    def aggregate_facial_features(self, frames: list[Frame]):
        feature_names = [
            two_landmark_connector.name
            for two_landmark_connector in frames[0].two_landmarks_connectors
        ]

        # Initialize feature lists
        feature_lists = {name: [] for name in feature_names}
        extra_features = {"smile": [], "pitch": [], "yaw": [], "roll": []}

        # Collect data from frames
        for frame in frames:
            if frame.smile is not None:
                extra_features["smile"].append(frame.smile)
            if frame.face_angles:
                extra_features["pitch"].append(frame.face_angles[0])
                extra_features["yaw"].append(frame.face_angles[1])
                extra_features["roll"].append(frame.face_angles[2])
            if frame.two_landmarks_connectors:
                for connector in frame.two_landmarks_connectors:
                    if connector.name in feature_lists:
                        feature_lists[connector.name].append(connector.length)

        # Aggregation functions
        agg_funcs = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "median": np.median,
        }

        # Compute statistics dynamically
        aggregated_features = {
            f"{key}_{agg_name}": agg_func(values)
            for key, values in {**feature_lists, **extra_features}.items()
            for agg_name, agg_func in agg_funcs.items()
        }

        return FacialFeatures(**aggregated_features)
