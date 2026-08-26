# Multimodal Fall Detection

Skeleton-based fall detection: classifies short video clips as **Fall** or **No Fall** (ADL) from human pose keypoints extracted with YOLOv8-Pose.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-purple?logo=ultralytics&logoColor=white)

## Motivation

A missed fall matters most for someone living alone: no one's there to notice, and the injury has time to get worse before anyone checks in. The hard part of this problem isn't spotting that a body is horizontal. It's telling apart a person who fell from a person who lay down on purpose, since both produce nearly identical skeleton poses. Posture alone is often ambiguous enough that a second signal, like the sound of an impact, is worth pursuing (see Limitations for why the audio branch stayed exploratory).

## Results

Evaluated on a held-out, deduplicated split of the training dataset (FallVision, 308 test videos):

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Fall         | 0.91      | 0.89   | 0.90     |
| No Fall      | 0.92      | 0.93   | 0.92     |
| **Accuracy** |           |        | **0.91** |

Fall recall (0.89) is the metric that matters most: a missed fall is far more costly than a false alarm. But a single held-out split from the training distribution only proves the model learned FallVision, not that it generalizes. See [`RESULTS.md`](RESULTS.md) for a second evaluation on GMDCSA24, a completely separate fall-detection dataset with different subjects and cameras: fall recall there drops to 0.66, and that gap is the more honest number.

## Key engineering findings

**Caught a data leak that was inflating accuracy to 98%.** Of 5,440 keypoint files in the training dataset, only 1,539 turned out to be unique by MD5 content hash, 72% were byte-identical duplicates scattered across folders. A random train/test split put copies of the same file on both sides, so the model was partly being graded on data it had memorized. Deduplicating by content hash before splitting dropped the reported accuracy from a misleading 98% to a real 91%.

**Caught a latent double-normalization bug before it reached a trained model.** The skeleton-centering and scaling transform originally lived only inside the training `Dataset` (`FallDataset`'s `transform` argument), not in `process_one_video` itself; `predict.py` calls `process_one_video` directly and never touches `Dataset`, so its inputs were normalized correctly the whole time. Later I moved the same transform into `process_one_video` so preprocessing is self-contained, which meant `data.py`'s `get_dataloaders` would have started applying it twice (once inside `process_one_video`, again via `Dataset`) had I trained again without changing anything else. I caught this by reading the two code paths side by side, and set `Dataset`'s `transform` to `None` before it ever produced a checkpoint. No model was retrained in between, so neither the FallVision 91% nor the GMDCSA24 result below was ever generated from a mismatched or double-normalized input; the bug is a real one, but it never shipped a bad number.

**Ran a cross-dataset generalization test, which most similar projects skip.** Testing on a second, independently collected dataset (GMDCSA24) is what actually surfaces domain shift instead of hiding behind a single in-distribution number. See [`RESULTS.md`](RESULTS.md).

## Approach

**Pipeline:** video &rarr; YOLOv8-Pose keypoint extraction (17 COCO joints per frame) &rarr; per-video cleanup &rarr; CNN classifier.

**Preprocessing** (`preprocess.py`), per video:
1. Deduplicate skeletons within a frame. Some frames have more than one detected skeleton (a real person plus a low-confidence phantom detection); keep the one with the highest average confidence.
2. Reshape the long-format keypoint CSV into a `(frames, 17, 3)` array: 17 joints, each with x, y, and detection confidence.
3. Resample every video to a fixed 64 frames via linear interpolation, so variable-length clips all end up the same shape.
4. Normalize: center each skeleton on the mid-hip point (removes absolute position) and scale to roughly a unit range.

**Model** (`Fall2d` in `helper_functions.py`): a small 2D CNN that treats each clip as a `(3, 64, 17)` image, 3 channels for x/y/confidence, 64 frames as height, 17 joints as width, so its filters pick up motion patterns across neighboring frames and joints. Two conv blocks, max pooling, dropout, and a linear classifier.

**Regularization:** on the deduplicated dataset the model overfit hard. Input scaling, dropout, and on-the-fly Gaussian-noise augmentation of joint coordinates during training brought the train/test gap down from about 12 points to about 3.

## Limitations

- No subject IDs in the source dataset, so a subject-independent split wasn't possible. The same person may show up in both train and test sets, which likely makes the FallVision numbers somewhat optimistic. This applies equally to any model trained on this data, so relative comparisons still hold.
- After deduplication, about 1,539 unique videos remain. That's not a lot, and it's why regularization mattered so much.
- The source dataset skews young and male, which may not represent the elderly population this system is actually meant for.
- I tried a YAMNet-based audio classifier (`audio.py`) to catch distress sounds (screams, impact thuds) as a second signal for the ambiguous cases. The test set was 7 clips, only one of which was a real recording of someone falling (the rest were me imitating what a fall might sound like), so it's nowhere near enough to draw a real conclusion. The one genuine recording did get flagged: YAMNet's top label for it was "Groan" (0.42) rather than "Screaming," but "Groan" is one of the distress keywords the fusion logic watches for, so it still would have tripped the filter. That's a promising anecdote, not evidence, which is why the audio branch stayed exploratory instead of getting fused into the main pipeline.

## Future work

- Collect a real (or at least larger and more representative) distress-audio dataset before deciding whether the audio branch is worth building out.
- Velocity/acceleration features derived from consecutive frames, not just static joint positions per frame.
- A graph-based model (ST-GCN), which encodes the actual skeletal connectivity instead of treating joints as adjacent pixels in an image.
- Recall-oriented threshold tuning and class-weighted loss, trading some precision for fewer missed falls, which is the right tradeoff for a safety system.

## Repository structure

| File | Role |
|---|---|
| `preprocess.py` | Per-video cleaning: dedup, reshape, resample, normalize. |
| `data.py` | Dataset assembly: content-hash dedup, train/test split, augmentation, PyTorch `Dataset`/`DataLoader`. |
| `helper_functions.py` | `Fall2d` model definition, video-to-keypoint-CSV extraction (YOLOv8-Pose), and the CSV-to-label inference helper. |
| `train.py` | Training loop, best-checkpoint saving, confusion matrix and classification report on the FallVision test split. |
| `predict.py` | Runs the trained model against a folder of new videos (used for the GMDCSA24 cross-dataset test in `RESULTS.md`). |
| `test_custom_csv.py` | Quick single-file sanity check against a saved checkpoint. |
| `audio.py` | YAMNet-based distress-sound classifier, exploratory, not part of the final pipeline (see Limitations). |
| `check_dup.py` | Standalone script that reports the content-hash duplication rate in the training data. |

## Running it yourself

Data is not included in this repository. To run `predict.py` on your own videos, organize them into two folders by class, for example `Fall/` and `ADL/`, each containing `.mp4` or `.mov` clips.

```bash
pip install torch ultralytics pandas numpy scikit-learn matplotlib tqdm mlxtend
python predict.py
```

Edit the `datafolder`, `fall_folder_name`, and `nofall_folder_name` arguments at the bottom of `predict.py` to point at your data. The script extracts keypoints with YOLOv8-Pose, runs the trained CNN (`best_model_test_91acc.pth`), and prints a confusion matrix and classification report.

## Datasets

Training data: FallVision. Harvard Dataverse. DOI: [10.7910/DVN/75QPKK](https://doi.org/10.7910/DVN/75QPKK).

Cross-dataset evaluation: E. Alam, A. Sufian, P. Dutta, M. Leo, I. A. Hameed, "GMDCSA24: A Dataset for Human Fall Detection in Videos," *Data in Brief* (2024). DOI: [10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216).
