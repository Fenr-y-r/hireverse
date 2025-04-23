import concurrent
import papermill as pm
import re
import os
from pathlib import Path
def get_p_and_pp_participant_number():


# Get the current script location (feature_extractor.py)
    current_file = Path(__file__).resolve()

# Go up to project root: HIREVERSE/
    project_root = current_file.parents[2]

# Build full path to the video file
    video_path = project_root / "data" / "raw" / "videos"

    print("Video path:", video_path)

# You can check if it exists
    if video_path.exists():
        print("Video file found!")
        VIDEOS_FOLDER = video_path
    else:
        print("Video file not found.")

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
        analyzer_paths = str(Path("./src/utils").resolve()),
        VidFile = str(Path("./data/raw/videos").resolve()),
        AudFile = str(Path("./data/raw/audio").resolve()),
        OutFile = str(Path("./data/processed/interview_features.csv").resolve())
        ),
        progress_bar=True
    )


p_participant_numbers, pp_participant_numbers = get_p_and_pp_participant_number()
execute_notebook("P1")
participant_ids = get_participant_ids(p_participant_numbers, pp_participant_numbers)
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(execute_notebook, participant_ids)
    for result in results:
        print(result)
