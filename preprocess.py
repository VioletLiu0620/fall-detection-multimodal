from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt
import random 

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

first = X[0]
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

frame_unique = [csv["Frame"].nunique() for csv in X]
# dict_frame = Counter(frame_unique)
# dict_frame_sort = dict(sorted(dict_frame.items()))

# plt.plot(list(dict_frame_sort.keys()), list(dict_frame_sort.values()))
# plt.show()

## Figure out the max frame could be 660, min could be 0, huge difference

# print(max(frame_unique))
# print(min(frame_unique))

## We need a pipeline to clean this data to the same shape N= 64 frames (64,17,3) (frames, skeleton points (X, Y, Confidence Score))

# df = pd.DataFrame(first)

# frame_dict = df.groupby("Frame")
# print(frame_dict.get_group(70))

# frame_size = frame_dict.size()
# bad_frame_index = frame_size[frame_size>17].index.to_list()

# for idx in bad_frame_index:
#     bad_frame_group = frame_dict.get_group(idx)
#     bad_frame_group["skeleton_id"] = bad_frame_group.groupby("Keypoint").cumcount()

#     confidence_mean = bad_frame_group.groupby(["Frame", "skeleton_id"])["Confidence"].mean()
#     best_frame, best_skeleton_id = confidence_mean.idxmax() # Returns a tuple of best_frame, best_skeleton_id, we only want id
#     print(best_frame, best_skeleton_id)


def dedup(video_csv: list):
    # Turn csv into a DataFrame
    df = pd.DataFrame(video_csv)

    # Add a column of skeleton id
    df["skeleton_id"] = df.groupby("Keypoint").cumcount()

    # Find the confidence mean across Frame and Skeleton ID
    df["skeleton_conf"] = df.groupby(["Frame", "skeleton_id"])["Confidence"].transform("mean")

    # Find out the max confidence in the group of Frame and Skeleton ID
    df["best_conf"] = df.groupby("Frame")["skeleton_conf"].transform("max")

    # Preserve the frame group with the highest conf
    df_clean = df[df["best_conf"] == df["skeleton_conf"]].copy()

    return df_clean

# Test dedup function whether it works on a single video frames (csv file)

random_int = random.randint(0, len(X))
clean = dedup(X[random_int])
print(f"Numbers of rows: {clean.shape[0]}")
print(f"Is the shape of the df matched with the nunique frame: {clean.shape[0] / 17 == clean["Frame"].nunique()}")

# Reshape the clean dataframe to array (nunique_frame, 17, 3) -> 17: 17 skeleton points (e.g. Nose, Left shoulder); 3: X, Y, Confidence Score

def reshape(clean_frame):
    # We need to first check if the frame has no missing keypoint, meaning one frame should always map to 17 keypoinys
    frame_size =  clean_frame.groupby("Frame").size()
    assert (frame_size == 17).all(), (f"Some frames don't have exactly 17 joints: ",
                                      f"{frame_size[frame_size != 17].to_dict()}")

    df_sorted = clean_frame.sort_values("Frame")
    nunique_frame = clean_frame["Frame"].nunique()
    target_values = df_sorted[["X", "Y", "Confidence"]].values
    print(f"Number of unique frame: {nunique_frame}")
    return target_values.reshape(nunique_frame, 17, 3)

reshaped_array = reshape(clean)
print(f"Array:\n{reshaped_array}")
print(f"Size after reshaped: {reshaped_array.shape}")

# Need another preprocess function that takes up our cleaned dataframe and size up/or down to N=64

def resample(reshaped_array, N=64):
    num_frames = reshaped_array.shape[0]
    old_times = np.linspace(0, 1, num_frames)
    new_times = np.linspace(0, 1, N)
    resampled = np.zeros((N, 17, 3))

    for keypoint in range(17):
        for coord in range(3):
            resampled[:, keypoint, coord] = np.interp(new_times, old_times, reshaped_array[:, keypoint, coord])

            # We compute all 64 frames all at once for a keypoint in coords
            # For example:
            # for frame 1-64, np.interp compute Nose at X, Y, Confidence in order
            # From the original array (reshaped_array), we know how Nose move from the first frame to last frame
            # np.interp know whether we need to size up or down by looking at the step difference in old_times vs new times (num_frames vs N)
            # Therefore, in order to fit into the new array (resampled), we re-compute Nose's movement togeteher
            # We first loop through Nose's X -> Nose's Y -> Nose's Confidence (fill out all 64 frames)
            # Then we change the keypoint, from Nose to left eye (left eye's X, Y, Confidence) etc.

    print(f"Size of resampled array: {resampled.shape}")

    return resampled

resampled = resample(reshaped_array= reshaped_array, N=64)

# # find indices of videos with fewer than 64 distinct frames
# short_indices = [i for i, count in enumerate(frame_unique) if count < 64]

# print(f"found {len(short_indices)} videos under 64 frames")
# print("first few:", short_indices[:5])

# short_video = X[short_indices[0]]
# clean = dedup(short_video)
# reshaped = reshape(clean)
# print(f"before resample: {reshaped.shape}")   # e.g. (45, 17, 3)

# resampled = resample(reshaped, N=64)
# print(f"after resample: {resampled.shape}")    # should be (64, 17, 3)