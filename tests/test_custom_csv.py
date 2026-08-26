"""
Quick sanity check: load a single custom keypoint CSV, run it through the
saved checkpoint, and print the predicted label. Not a formal test suite,
just a fast manual check that the pipeline + a checkpoint still agree.
"""
import sys
from pathlib import Path

# src/ isn't a package; this makes preprocess.py and helper_functions.py
# importable when this script is run directly (e.g. `python tests/test_custom_csv.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pandas as pd
from preprocess import process_one_video
from helper_functions import Fall2d

REPO_ROOT = Path(__file__).resolve().parent.parent
csv_path = REPO_ROOT / "custom_data" / "c_f_c_a.csv"
video_df = pd.read_csv(csv_path)
X = torch.from_numpy(process_one_video(video_csv=video_df)).float().unsqueeze(dim=0)

label_map = {"Fall": 0,
             "No Fall": 1}
class_names = list(label_map.keys())

model = Fall2d(input_shape=3,
               output_shape=2,
               hidden_units=16)

model.load_state_dict(torch.load(REPO_ROOT / "best_model_test_91acc.pth"))
model.eval()

with torch.inference_mode():
    y_logits = model(X)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).item()
    y_label = class_names[y_pred]

print(f"The model classified {csv_path.name} as {y_label}")
