# Skeleton-Based Fall Detection

Pose-based fall detection: classifies short video clips as **Fall** or **No Fall** (ADL) from human pose keypoints extracted with YOLOv8-Pose. The trained model uses skeleton motion only; an audio branch was explored but deliberately not integrated (see "Audio Exploration" below).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-purple?logo=ultralytics&logoColor=white)

## Motivation

A missed fall matters most for someone living alone: no one's there to notice, and the injury has time to get worse before anyone checks in. The hard part of this problem isn't spotting that a body is horizontal. It's telling apart a person who fell from a person who lay down on purpose, since both produce nearly identical skeleton poses. Posture alone is often ambiguous enough that a second signal, like the sound of an impact, seemed worth testing, which is why an audio classifier was built and evaluated even though it didn't make it into the final system.

## Results

Evaluated on a held-out, deduplicated split of the training dataset (FallVision, 308 test videos):

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Fall         | 0.91      | 0.89   | 0.90     |
| No Fall      | 0.92      | 0.93   | 0.92     |
| **Accuracy** |           |        | **0.91** |

Fall recall (0.89) is the metric that matters most: a missed fall is far more costly than a false alarm. But a single held-out split from the training distribution only proves the model learned FallVision, not that it generalizes. See [`RESULTS.md`](RESULTS.md) for a second evaluation on GMDCSA24, a completely separate fall-detection dataset with different subjects and cameras: fall recall there drops to 0.66, and that gap is the more honest number.

## Key engineering findings

- Deduplicating the training files exposed a data leak that inflated accuracy to 98%. Out of 5,440 keypoint files in FallVision, only 1,539 were unique. The remaining 72% were byte-identical duplicates scattered across folders. A standard random train/test split placed identical copies on both sides, letting the model memorize files. Cleaning out duplicate hashes brought the real accuracy to 91%.
- Inspecting the training pipeline caught a double-normalization bug before training ruined a checkpoint. The spatial scaling transform was running inside the `Dataset` class, but `process_one_video` was also scaling inputs directly. Had the dataset been reassembled without setting `Dataset` transform parameters to `None`, inputs would have been normalized twice.
- Testing across datasets revealed heavy domain shift. Evaluating on GMDCSA24 dropped fall recall from 0.89 to 0.66, showing how badly single-dataset benchmarks mask real-world degradation.

## Approach

Video clips pass through YOLOv8-Pose to extract 17 COCO keypoints per frame before entering a 2D CNN classifier (`Fall2d` in `src/helper_functions.py`).

Per-video preprocessing in `src/preprocess.py` cleans and formats spatial data through four steps:
1. Skeletons within a single frame are deduplicated by picking the detection with the highest average confidence score.
2. Long-format CSV keypoints reshape into a `(frames, 17, 3)` tensor tracking 17 joints across x, y, and confidence values.
3. Linear interpolation resamples every video to 64 frames to standardize clip length.
4. Joints center on the mid-hip point and scale to unit range to remove position bias.

The classifier treats the clip as a `(3, 64, 17)` tensor, mapping 3 channels across 64 frames (height) and 17 joints (width). Two convolutional blocks, max pooling, dropout, and a final linear layer learn spatial motion patterns across adjacent joints and frames.

Because the model overfit early on the smaller deduplicated dataset, regularization was applied during training. Input scaling, dropout, and on-the-fly Gaussian noise added to joint coordinates brought the train/test performance gap down from 12 percentage points to 3.

## Audio Exploration (Not Integrated)

Posture alone is ambiguous for the hardest cases (a gentle fall versus lying down), so a YAMNet-based distress-sound classifier (`exploration/audio.py`) was built to test whether audio could resolve them: extract each clip's audio, run it through YAMNet, and check whether fall-relevant classes (scream, groan, thud, crying) show up as top predictions.

The test set was 7 clips, only one of which was a real recording of someone falling; the rest were self-recorded imitations of what a fall might sound like. That's nowhere near enough to validate a classifier on. The one genuine recording did get flagged: YAMNet's top label for it was "Groan" (0.42) rather than "Screaming," but "Groan" is one of the distress keywords the detection logic watches for, so it still would have tripped the filter. That's a promising anecdote, not evidence.

Two things kept this out of the main pipeline rather than pushed toward fusion: the evaluation set is too small and too synthetic to know if it generalizes past luck, and most real home-monitoring setups don't reliably capture audio anyway, which caps how much value an audio branch adds even if it worked. Building something and then deciding not to ship it, with the evidence for that decision written down, felt more honest than fusing in a classifier validated on 7 clips because it was already built. If a larger, more representative distress-audio dataset becomes available, this is the first thing to revisit (see Future work).

## Limitations

- No subject IDs in the source dataset, so a subject-independent split wasn't possible. The same person may show up in both train and test sets, which likely makes the FallVision numbers somewhat optimistic. This applies equally to any model trained on this data, so relative comparisons still hold.
- After deduplication, about 1,539 unique videos remain. That's not a lot, and it's why regularization mattered so much.
- The source dataset skews young and male, which may not represent the elderly population this system is actually meant for.
- The audio branch stayed exploratory rather than validated; the model shipped here is skeleton-only.

## Future work

- Collect a real distress-audio dataset before deciding whether the audio branch is worth building out.
- Derive velocity and acceleration features from consecutive frames instead of relying solely on static joint positions per frame.
- Switch to a graph-based model (ST-GCN) to encode true skeletal connectivity instead of treating joints as adjacent pixels in an image grid.
- Tune thresholds for recall and apply class-weighted loss to trade some precision for fewer missed falls.

## Repository structure

```
fall-detection-multimodal/
├── src/                    # core pipeline
│   ├── preprocess.py       # per-video cleaning: dedup, reshape, resample, normalize
│   ├── data.py              # dataset assembly: content-hash dedup, split, augmentation, DataLoaders
│   ├── helper_functions.py # Fall2d model, video-to-keypoint-CSV extraction, inference helper
│   ├── train.py             # training loop, best-checkpoint saving, evaluation on FallVision
│   └── predict.py           # run the trained model against a folder of new videos
├── scripts/                 # one-off utilities, not imported by the pipeline
│   ├── check_dup.py         # reports content-hash duplication rate in the training data
│   ├── open_rar.py          # extracts a downloaded .rar of keypoint CSVs into data/
│   └── mov_to_label.py      # stub, not implemented (see file header)
├── exploration/              # not part of the shipped pipeline
│   ├── audio.py              # YAMNet distress-sound classifier (see Audio Exploration above)
│   └── yamnet_class_map.csv  # YAMNet's class label reference, needed by audio.py
├── tests/
│   └── test_custom_csv.py    # quick manual sanity check against a saved checkpoint
├── assets/
│   └── confusion_matrix.png  # GMDCSA24 confusion matrix, referenced from RESULTS.md
├── best_model.pth / best_model_test_91acc.pth   # saved checkpoints
├── README.md
└── RESULTS.md                # cross-dataset generalization writeup
```

## Running it yourself

Data is not included in this repository. To run `predict.py` on your own videos, organize them into two folders by class, for example `Fall/` and `ADL/`, each containing `.mp4` or `.mov` clips.

```bash
pip install torch ultralytics pandas numpy scikit-learn matplotlib tqdm mlxtend
python src/predict.py
```

Run this from the repository root (checkpoints and the YOLO weights are loaded by relative path). Edit the `datafolder`, `fall_folder_name`, and `nofall_folder_name` arguments at the bottom of `src/predict.py` to point at your data. The script extracts keypoints with YOLOv8-Pose, runs the trained CNN (`best_model_test_91acc.pth`), and prints a confusion matrix and classification report.

## Datasets

Training data: FallVision. Harvard Dataverse. DOI: [10.7910/DVN/75QPKK](https://doi.org/10.7910/DVN/75QPKK).

Cross-dataset evaluation: E. Alam, A. Sufian, P. Dutta, M. Leo, I. A. Hameed, "GMDCSA24: A Dataset for Human Fall Detection in Videos," *Data in Brief* (2024). DOI: [10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216).