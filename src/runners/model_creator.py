import concurrent
import os
import nbformat
import pandas as pd
import papermill as pm


def execute_notebook(label, drop_some_facial_features=False):
    current_dir = os.path.dirname(__file__)
    output_file_path = os.path.join(
        current_dir, "runners", "outputs", f"{label}_runner_output.ipynb"
    )
    model_dir = os.path.join(current_dir, "hirability_model.ipynb")
    pm.execute_notebook(  # TODO: use relative path here and other runner
        model_dir,
        output_file_path,
        parameters=dict(target_column=label),
        progress_bar=False,
    )
    return get_scores(label)

def get_scores(label):
    current_dir = os.path.dirname(__file__)
    notebook_path = os.path.join(
        current_dir, "runners", "outputs", f"{label}_runner_output.ipynb"
    )

    with open(notebook_path, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Initialize variables to store the values
    avg_r2_score = None
    avg_pearson_score = None

    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code":
            # Check if avg_r2_score or avg_pearson_score is calculated in this cell
            if "avg_r2_score" in cell["source"]:
                # Look for the value of avg_r2_score in the output
                for output in cell.get('outputs', []):
                    if 'data' in output:
                        if "text/plain" in output.data:
                            avg_r2_score = output.data["text/plain"].strip()

            if "avg_pearson_score" in cell["source"]:
                # Look for the value of avg_pearson_score in the output
                for output in cell.get('outputs', []):
                    if 'data' in output:
                        if "text/plain" in output.data:
                            avg_pearson_score = output.data["text/plain"].strip()

    # Return a dictionary with label, avg_r2_score, and avg_pearson_score
    return {"label": label, "avg_r2_score": avg_r2_score, "avg_pearson_score": avg_pearson_score}


script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "datasets", "turker_scores_full_interview.csv")
df = pd.read_csv(file_path)
labels = df.columns[3:].tolist()
list = []
for label in labels:
    list.append(execute_notebook(label))
    print(list)
