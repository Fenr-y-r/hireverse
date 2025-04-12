import concurrent
import os
import re
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


import os
import nbformat
import re
from bs4 import BeautifulSoup


def get_scores(label):
    current_dir = os.path.dirname(__file__)
    notebook_path = os.path.join(
        current_dir, "runners", "outputs", f"{label}_runner_output.ipynb"
    )

    if not os.path.exists(notebook_path):
        return {"label": label, "avg_r2_score": None, "avg_pearson_score": None}

    with open(notebook_path, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    avg_r2_score = None
    avg_pearson_score = None

    for cell in notebook_content["cells"]:
        if cell["cell_type"] == "code":
            for output in cell.get("outputs", []):
                output_text = ""
                # Handle "stream"/"execute_result"/other outputs
                if "data" in output:
                    for mime_type in ["text/plain", "text/html", "text/latex"]:
                        if mime_type in output.data:
                            if mime_type == "text/html":
                                soup = BeautifulSoup(
                                    output.data[mime_type], "html.parser"
                                )
                                output_text += soup.get_text(separator=" ").strip()
                            else:
                                output_text += output.data[mime_type].strip()
                            output_text += "\n"
                elif "text" in output:
                    output_text = output["text"].strip()

                # Match scores
                r2_match = re.search(
                    r"Mean R² Score[:\s]*([0-9.]+)(\s*\(±[0-9.]+\))?",
                    output_text,
                    re.IGNORECASE,
                )
                pearson_match = re.search(
                    r"Mean Pearson Correlation[:\s]*([0-9.]+)(\s*\(±[0-9.]+\))?",
                    output_text,
                    re.IGNORECASE,
                )

                if r2_match and not avg_r2_score:
                    avg_r2_score = r2_match.group(1)
                if pearson_match and not avg_pearson_score:
                    avg_pearson_score = pearson_match.group(1)

    return {
        "label": label,
        "avg_r2_score": avg_r2_score,
        "avg_pearson_score": avg_pearson_score,
    }


script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "datasets", "turker_scores_full_interview.csv")
df = pd.read_csv(file_path)
labels = df.columns[3:].tolist()
list = []
for label in labels:
    list.append(execute_notebook(label))
    sorted_data = sorted(list, key=lambda x: x["avg_pearson_score"], reverse=True)
    print(sorted_data)

# [
#     {"label": "Excited", "avg_r2_score": "0.34", "avg_pearson_score": "0.64"},
#     {"label": "EngagingTone", "avg_r2_score": "0.30", "avg_pearson_score": "0.62"},
#     {"label": "Smiled", "avg_r2_score": "0.28", "avg_pearson_score": "0.60"},
#     {"label": "Friendly", "avg_r2_score": "0.20", "avg_pearson_score": "0.55"},
#     {"label": "RecommendHiring", "avg_r2_score": "0.20", "avg_pearson_score": "0.54"},
#     {"label": "StructuredAnswers", "avg_r2_score": "0.23", "avg_pearson_score": "0.54"},
#     {"label": "Engaged", "avg_r2_score": "0.14", "avg_pearson_score": "0.50"},
#     {"label": "Total", "avg_r2_score": "0.15", "avg_pearson_score": "0.50"},
#     {"label": "NoFillers", "avg_r2_score": "0.15", "avg_pearson_score": "0.48"},
#     {"label": "Colleague", "avg_r2_score": "0.14", "avg_pearson_score": "0.47"},
#     {"label": "Paused", "avg_r2_score": "0.15", "avg_pearson_score": "0.47"},
#     {"label": "Calm", "avg_r2_score": "0.09", "avg_pearson_score": "0.43"},
#     {"label": "NotAwkward", "avg_r2_score": "0.07", "avg_pearson_score": "0.43"},
#     {"label": "SpeakingRate", "avg_r2_score": "0.08", "avg_pearson_score": "0.42"},
#     {"label": "Focused", "avg_r2_score": "0.03", "avg_pearson_score": "0.37"},
#     {"label": "Authentic", "avg_r2_score": "0.00", "avg_pearson_score": "0.34"},
#     {"label": "NotStressed", "avg_r2_score": None, "avg_pearson_score": "0.30"},
#     {"label": "EyeContact", "avg_r2_score": None, "avg_pearson_score": "0.28"},
# ]