import concurrent
from hireverse.utils.utils import BASE_DIR, get_participant_ids
import papermill as pm  
import re
import os
from pathlib import Path

def execute_notebook(participant_id):
    print(participant_id)
    input_path = os.path.join(BASE_DIR, "src","hireverse","pipelines", "feature_extractor.ipynb")
    output_path =os.path.join(BASE_DIR,  "outputs", "feature_extractor_output.ipynb")

    pm.execute_notebook(
        input_path=input_path,
        output_path=output_path,
        parameters=dict(
            participant_id=participant_id,
        ),
        progress_bar=True
    )


participant_ids = get_participant_ids()
participant_ids.remove("P13")
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(execute_notebook, participant_ids)
