"""
One-off utility: extract a downloaded .rar of keypoint CSVs into data/.
Run from anywhere; paths are resolved relative to the repo root, not
hardcoded to any one machine.
"""
from pathlib import Path
import rarfile

REPO_ROOT = Path(__file__).resolve().parent.parent
RAR_PATH = REPO_ROOT / "data" / "f_mask_b_1_keypoints_csv.rar"
EXTRACT_TO = REPO_ROOT / "data"

with rarfile.RarFile(RAR_PATH) as rf:
    rf.extractall(path=EXTRACT_TO)
