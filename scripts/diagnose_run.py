"""
Run-level diagnostic. Mirrors predict.py's loop exactly, but logs what each
step actually paired together, so the two machines can be diffed row by row.

    python scripts/diagnose_run.py "<datafolder>" Fall ADL

Writes diagnose_run_<host>.csv next to the repo root and prints the table.
"""
import sys, platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from helper_functions import mov_to_csv, Fall2d
from preprocess import process_one_video

REPO_ROOT = Path(__file__).resolve().parent.parent
datafolder = Path(sys.argv[1])
fall_name = sys.argv[2] if len(sys.argv) > 2 else "Fall"
nofall_name = sys.argv[3] if len(sys.argv) > 3 else "ADL"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Fall2d(input_shape=3, output_shape=2, hidden_units=16)
model(torch.randn(1, 64, 17, 3))
ckpt = REPO_ROOT / "best_model_test_91acc.pth"
sd = torch.load(ckpt, map_location="cpu")
model.load_state_dict(sd)
model.to(device).eval()

# fingerprint the checkpoint so we can prove both machines loaded the same weights
flat = torch.cat([v.flatten().double() for v in sd.values()])
print(f"device={device}  checkpoint={ckpt.name}  n_params={flat.numel()}  "
      f"sum={flat.sum().item():.6f}  absmean={flat.abs().mean().item():.8f}")
print(f"cwd={Path.cwd()}")

video_list, y_trues = [], []
for video in sorted(list(datafolder.rglob("*"))):
    if video.suffix.lower() in [".mp4", ".mov"]:
        video_list.append(video)
        y_trues.append(video.parent.name)
print(f"videos found: {len(video_list)}  labels: {sorted(set(y_trues))}")

label_map = {0: fall_name, 1: nofall_name}
rows = []
for i, (video, y_true) in enumerate(zip(video_list, y_trues)):
    csv_path = mov_to_csv(video)
    df = pd.read_csv(csv_path)
    arr = process_one_video(video_csv=df)
    X = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(X)[0]
    pred = label_map[int(torch.argmax(logits))]
    rows.append(dict(
        i=i,
        video=str(video),
        y_true=y_true,
        csv=str(csv_path),
        # does the csv the loop read actually belong to this video?
        csv_matches_video=(Path(csv_path).stem == video.stem
                           and Path(csv_path).parent == video.parent),
        rows=len(df), frames=int(df["Frame"].nunique()),
        xy_max=round(float(np.abs(arr[:, :, :2]).max()), 3),
        pred=pred,
        margin=round(float(logits[0] - logits[1]), 3),
    ))

out = pd.DataFrame(rows)
host = platform.system().lower() + ("_cuda" if device == "cuda" else "_cpu")
out.to_csv(REPO_ROOT / f"diagnose_run_{host}.csv", index=False)
print(out.to_string())
print(f"\nmismatched csv/video pairs: {(~out.csv_matches_video).sum()}")
print(f"unique csv files read: {out.csv.nunique()} (should equal {len(out)})")
print(f"\naccuracy: {(out.y_true == out.pred).mean():.4f}")
print(confusion_matrix(out.y_true, out.pred, labels=[fall_name, nofall_name]))
print(classification_report(out.y_true, out.pred,
                            labels=[fall_name, nofall_name],
                            target_names=[fall_name, nofall_name]))
print(f"\nwrote diagnose_run_{host}.csv")
