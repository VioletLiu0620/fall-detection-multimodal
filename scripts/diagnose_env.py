"""
Run this on BOTH machines (local CPU and Colab GPU) and diff the two outputs.

It isolates *where* the two runs diverge: the CSV that YOLO produces, the
array that preprocess.py builds from it, or the logits the classifier returns.
Point it at one video whose local margin is known (ADL/05.mp4 -> -20.12).

    python scripts/diagnose_env.py "<path to a video>.mp4"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import ultralytics, cv2

from helper_functions import mov_to_csv, Fall2d
from preprocess import dedup, reshape, resample, custom_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
video = Path(sys.argv[1])

print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | cv2 {cv2.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")

cap = cv2.VideoCapture(str(video))
print(f"decoded video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, orientation meta {cap.get(cv2.CAP_PROP_ORIENTATION_META)}")

csv_path = mov_to_csv(video)
print(f"csv actually written to: {csv_path}  (exists: {Path(csv_path).exists()})")

df = pd.read_csv(csv_path)
print(f"rows {len(df)}, distinct frames {df['Frame'].nunique()}")
# The single most important line: raw keypoints must be in PIXELS of the decoded
# frame. If X/Y max out near 1.0, or near 640, the /200.0 in custom_transform is
# operating on the wrong scale and every prediction collapses toward "Fall".
print("raw X range: %.2f .. %.2f | raw Y range: %.2f .. %.2f"
      % (df.X.min(), df.X.max(), df.Y.min(), df.Y.max()))
print("confidence quantiles (0,25,50,75,100): "
      + str(np.quantile(df.Confidence, [0, .25, .5, .75, 1]).round(3)))

arr = reshape(dedup(df))
transformed = custom_transform(resample(arr))
print(f"model input shape {transformed.shape}, "
      f"|xy| max {np.abs(transformed[:, :, :2]).max():.3f} "
      f"(expect roughly 1-4; >>5 means the scale is wrong)")

model = Fall2d(input_shape=3, output_shape=2, hidden_units=16)
model(torch.randn(1, 64, 17, 3))
model.load_state_dict(torch.load(REPO_ROOT / "best_model_test_91acc.pth", map_location="cpu"))
model.eval()

for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    model.to(device)
    X = torch.from_numpy(transformed).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(X)[0]
    print(f"{device}: logits {logits.tolist()}  margin(fall-adl) {float(logits[0] - logits[1]):.4f}")
