import concurrent
import os
import pandas as pd
import papermill as pm

def execute_notebook(label, drop_some_facial_features):
    pm.execute_notebook(    # TODO: use relative path here and other runner
        "../regression_models/hirability_model.ipynb",
        f"../regression_models/outputs/{label}.ipynb",
        parameters=dict(label=label, drop_some_facial_features= drop_some_facial_features),
        progress_bar=False,
    )

def get_scores(label):
    executed_notebook = pm.read_notebook("../regression_models/outputs/{label}.ipynb")

    final_accuracy = None
    
    for cell in executed_notebook.cells:
        if 'final_accuracy' in cell['metadata']:  # Check if the variable is in the cell metadata
            final_accuracy = cell['outputs'][0]['text/plain']
            break
    
    return final_accuracy


script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "..", "datasets", "turker_scores_full_interview.csv")
df = pd.read_csv(file_path)
labels = df.columns[1:].tolist()

for label in labels:
    execute_notebook(label)