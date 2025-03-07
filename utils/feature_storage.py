from dataclasses import asdict
import pandas as pd
import os

from models.prosodic_features import ProsodicFeatures


class FeatureStorage:
    CSV_PATH = "./interview_features.csv"

    @classmethod
    def save_to_csv(cls, participant_id, prosodic_features: ProsodicFeatures):

        data = {
            "participant_id": participant_id,
            **asdict(prosodic_features),  # Unpacks prosodic features
        }

        df = pd.DataFrame([data])

        if not os.path.exists(cls.CSV_PATH):  
            # Create new file if it doesn’t exist
            df.to_csv(cls.CSV_PATH, index=False)
        else:
            existing_df = pd.read_csv(cls.CSV_PATH)
            existing_ids = set(
                existing_df["participant_id"]
            )  # # Convert participant_id column to a set for O(1) lookup
            if participant_id in existing_ids:
                return  
            df.to_csv(
                cls.CSV_PATH, mode="a", header=False, index=False
            )  # Append new row
