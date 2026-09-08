# Cross-Dataset Generalization Test

The number in the main README comes from a held-out split of FallVision, the dataset the model was trained on. That shows the model learned FallVision. It doesn't show whether it generalizes to a new camera, a new room, or people it's never seen, which is what actually matters for a real deployment. So I ran the trained model against GMDCSA24, a fall detection dataset built by a different research group, with different subjects, cameras, and rooms, never seen during training or model selection.

## Setup

- Dataset: GMDCSA24 (Alam et al., 2024), 4 subjects, 160 videos, split close to evenly between Fall and ADL (Activities of Daily Living).
- Pose extraction: my own YOLOv8-pose pipeline (`mov_to_csv` in `helper_functions.py`), not GMDCSA24's own keypoint annotations. This tests the full pipeline end to end, extraction included, not just the classifier on pre-cleaned input.
- Model: `best_model.pth`, trained on FallVision, no fine-tuning and no retraining on GMDCSA24.

## Results

![Confusion matrix on GMDCSA24](assets/confusion_matrix.png)

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Fall         | 0.80      | 0.76   | 0.78     | 79      |
| ADL          | 0.78      | 0.81   | 0.80     | 81      |
| **Accuracy** |           |        | **0.79** | 160     |
| Macro avg    | 0.79      | 0.79   | 0.79     | 160     |
| Weighted avg | 0.79      | 0.79   | 0.79     | 160     |

Confusion matrix (rows true, columns predicted):

|          | pred Fall | pred ADL |
|----------|-----------|----------|
| **Fall** | 60        | 19       |
| **ADL**  | 15        | 66       |

### Per subject

Accuracy varies far more by subject than the pooled number suggests:

| Subject | Videos | Fall recall | ADL recall | Accuracy |
|---------|--------|-------------|------------|----------|
| 1       | 32     | 0.94        | 0.94       | **0.94** |
| 2       | 48     | 0.72        | 0.87       | 0.79     |
| 3       | 43     | 0.67        | 0.86       | 0.77     |
| 4       | 37     | 0.76        | 0.60       | **0.68** |

## Reading the numbers

Overall accuracy is 0.79, and 60 of 79 falls are caught. The pooled figure hides a 26-point spread between the best subject (0.94) and the worst (0.68), which is the real story: domain shift here is **per-person**, not per-dataset. Subject 4 is the only one where ADL recall collapses (0.60), meaning the model reads that person's ordinary activities as falls. Whatever differs about how subject 4 moves, sits, or is framed by the camera, the model has not seen it.

A missed fall is the costly error, so fall recall of 0.76 is the number to quote for real-world reliability, not the 0.9-something from the training distribution. Nineteen missed falls out of 79 would be nineteen real emergencies nobody was alerted to.

Most portfolio fall-detection projects report one clean test-set number and stop there. A single held-out split from the training dataset can only tell you how well the model memorized that dataset's distribution.

## The bug this evaluation caught

The first version of this evaluation scored 0.78 accuracy with 0.66 fall recall. Re-running the identical code on Colab scored **0.41** — barely better than a coin flip, with the same weights, the same videos, and the same checkpoint.

The cause was one line in `reshape()` (`src/preprocess.py`):

```python
df_sorted = clean_frame.sort_values("Frame")     # kind="quicksort" by default — not stable
```

Each frame contributes 17 rows, one per joint, all sharing the same `Frame` value. Sorting on `Frame` alone leaves all 17 tied, and an unstable sort is free to return ties in any order. The next line reshapes to `(frames, 17, 3)` assuming slot *k* holds joint *k*, and `custom_transform` reads slots 11 and 12 as the hips.

Measured over 5,858 frames, **zero** had the joints in the correct slots and **zero** had the hips at slots 11 and 12. The mid-hip centering had never once centered on a hip.

The shuffle is deterministic for a given numpy build but differs across builds — arm64 macOS and x86-64 Colab produce different permutations. The model had been trained through the same broken path, so it had memorized one machine's scramble; on any other machine it was reading noise. The diagnostic that pinned it: `ADL/16.mp4` produced identical detections on both machines (1768 rows, 104 frames) and scored −10.80 on one and +0.73 on the other.

The fix sorts on the joint name as well as the frame, using an ordered categorical so the result cannot depend on row order, pandas version, or CPU:

```python
clean_frame["Keypoint"] = pd.Categorical(clean_frame["Keypoint"], categories=joint_names, ordered=True)
df_sorted = clean_frame.sort_values(by=["Frame", "Keypoint"], ascending=[True, True])
```

After the fix, 100% of frames carry correctly ordered joints and `reshape` is provably invariant to input row order. Retrained on properly ordered skeletons:

| | before fix | after fix |
|---|---|---|
| FallVision held-out accuracy | 0.91 | 0.89 |
| FallVision fall recall | 0.89 | **0.90** |
| Subject 1 accuracy | 0.81 | **0.94** |
| GMDCSA24 accuracy (160) | 0.78 | 0.79 |
| GMDCSA24 fall recall | 0.66 | **0.76** |
| falls caught | 52 / 79 | **60 / 79** |
| same result on macOS and Colab | no (0.81 vs 0.41) | **yes** |

Note the in-distribution score went *down*, from 0.91 to 0.89. That is the expected shape of this fix: part of the old 0.91 was the model exploiting a machine-specific artifact of its own training data rather than learning posture. Trading two points of in-distribution accuracy for ten points of cross-dataset fall recall is the trade worth making.

Pooled accuracy barely moved, which is worth being honest about — but the error balance shifted in the direction that matters, trading 6 extra false alarms for 8 more falls caught. The larger result is that the number is now reproducible on any machine, which it previously was not.

## Reproducing this

```bash
python src/predict.py
```

Run from the repository root. Enter the path to a folder of GMDCSA24 videos organized into per-class subfolders, then the Fall and ADL folder names, when prompted.

## Dataset citation

E. Alam, A. Sufian, P. Dutta, M. Leo, I. A. Hameed, "GMDCSA24: A Dataset for Human Fall Detection in Videos," *Data in Brief* (2024). Dataset DOI: [10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216).
