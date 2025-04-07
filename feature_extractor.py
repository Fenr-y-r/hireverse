import concurrent
import papermill as pm
import re
import os

import os
import re


def get_p_and_pp_participant_number():
    VIDEOS_FOLDER = "./MIT/Videos/"
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
        "/Users/bassel27/personal_projects/hireverse/feature_extractor.ipynb",
        "/Users/bassel27/personal_projects/hireverse/lol.ipynb",
        parameters=dict(participant_id=participant_id),
        progress_bar=False,
    )


p_participant_numbers, pp_participant_numbers = get_p_and_pp_participant_number()

participant_ids = get_participant_ids(p_participant_numbers, pp_participant_numbers)
participant_ids.remove("P13")
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(execute_notebook, participant_ids)
    for result in results:
        print(result)
