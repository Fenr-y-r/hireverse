import concurrent
import papermill as pm  
import re
import os
from pathlib import Path


def get_p_and_pp_participant_number():
    current_folder = Path(__file__).resolve()
    project_root = current_folder.parents[2]
    VIDEOS_FOLDER = project_root / "data" / "raw" / "videos"
        

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


def get_participant_ids(p_participant_numbers, pp_participant_numbers):
    participant_ids = []
    for prefix, participant_numbers in [
        ("P", p_participant_numbers),
        ("PP", pp_participant_numbers),
    ]:
        for participant_number in participant_numbers:
            participant_id = f"{prefix}{participant_number}"
            participant_ids.append(participant_id)
    return participant_ids
 

def execute_notebook(participant_id):
    print(participant_id)
    pm.execute_notebook(
        input_path=Path(__file__).resolve().parent.parent / "pipelines"/ "feature_extractor.ipynb",
        output_path=Path(__file__).resolve().parent.parent.parent / "outputs"/ "feature_extractor_output.ipynb",
        parameters=dict(
        participant_id=participant_id,
        ),
        progress_bar=True
    )



p_participant_numbers, pp_participant_numbers = get_p_and_pp_participant_number()

participant_ids = get_participant_ids(p_participant_numbers, pp_participant_numbers)
participant_ids.remove("P13")
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(execute_notebook, participant_ids[:2])
