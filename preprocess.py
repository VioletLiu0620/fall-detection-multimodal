from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt

data_path = Path("data")
all_keypoints_list = list(data_path.glob("**/*.csv"))

print(f"Total numbers of video: {len(all_keypoints_list)}")

fall = sum(1 for p in all_keypoints_list if "No Fall" not in str(p))
no_fall = sum(1 for p in all_keypoints_list if "No Fall" in str(p))
print(f"fall: {fall}, no_fall: {no_fall}")

X = []
y = []

for each_csv in all_keypoints_list:
    X.append(pd.read_csv(each_csv))
    y.append(each_csv.parts[1])

## Find the problem of max frame != actual frames, there will be one frame with two sets of skeleton, and frame missing issue

# first = X[0]
# print(first)
# print(all_keypoints_list[0])
# print(f"Max frame: {first["Frame"].max()}")
# print(f"Actual frame: {first["Frame"].nunique()}")

## Check on frame number mistmatch issues

# messy = 0
# clean = 0
# for csv in all_keypoints_list:
#     df = pd.read_csv(csv)
#     n_rows = len(df)
#     n_frames = df["Frame"].nunique()
#     if n_rows == n_frames * 17:
#         clean += 1
#     else:
#         messy += 1
# print(f"clean: {clean}, messy: {messy}")

## Find a good number by visualizing how many distincts frames these video have

# frame_unique = [csv["Frame"].nunique() for csv in X]
# dict_frame = Counter(frame_unique)
# dict_frame_sort = dict(sorted(dict_frame.items()))

# plt.plot(list(dict_frame_sort.keys()), list(dict_frame_sort.values()))
# plt.show()

## Figure out the max frame could be 660, min could be 0, huge difference

# print(max(frame_unique))
# print(min(frame_unique))

## We need a pipeline to clean this data to the same shape N= 64 frames (64,17,3) (frames, skeleton points (X, Y, Confidence Score))

def dedup(video_frame: list):
    # Start with groupby in pandas
    return