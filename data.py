from preprocess import process_one_video
import torch
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import hashlib

## We need to normalize: Center on hip
def custom_transform(video_array):
    ar = video_array.copy()
    xy = ar[:, :, :2]
    # print(xy.shape)
    conf = ar[:, :, 2:]

    LEFT_HIP, RIGHT_HIP = 11, 12
    mid_hip = (xy[:, LEFT_HIP, :] + xy[:, RIGHT_HIP, :]) / 2 # midhip.shape = (64, 2)
    xy =  xy - mid_hip[:, None, :] # change midhip into (64, 1, 2) to be able minus xy
    xy = xy / 200.0                        # ← scale to roughly ±1 range
    result = np.concatenate((xy, conf), axis=2) # conf is joining at last dimenstion, xy: (64, 17, 2) + conf: (64, 17, 1) - > result: (64, 17, 3)

    return result

# first = X_train[0]
# print(f"Shape before normalize: {first.shape}")
# result = custom_transfrom(first)
# print(result)
# print(f"Shape after normalize: {result.shape}")

## Define some data augmentaion function

def add_noise(video_array, std: float):
    noise = np.random.normal(loc=0, scale=std, size=video_array[:, :, :2].shape)
    video_array = video_array.copy()
    video_array[:, :, :2] += noise
    return video_array

## Create custom dataset class
class FallDataset(Dataset):
    def __init__(self, X, y, transform= None, augmentation= False):
        self.X = X
        self.y = y
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        label = self.y[index]

        if self.transform:
            x = self.transform(x)

        if self.augmentation:
            if np.random.rand() > 0.5:
                x = add_noise(x, std= 0.03)

        # convert to tensor
        x = torch.tensor(x, dtype= torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return x, label

# read the file content in bytes, feed those bytes in md5 finger print machine then to hex readable string
def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def get_dataloaders(BATCH_SIZE=32, NUM_WORKERS=0):
    data_path = Path("data")
    all_keypoints_list = list(data_path.glob("**/*.csv"))

    print(f"Total numbers of video: {len(all_keypoints_list)}")

    unique_files = []
    seen_hashes = set() # set has no duplicates
    for csv in all_keypoints_list:
        h = file_hash(csv)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_files.append(csv)

    print(f"Number of unique csv: {len(unique_files)}")

    fall = sum(1 for p in unique_files if "No Fall" not in str(p))
    no_fall = sum(1 for p in unique_files if "No Fall" in str(p))
    print(f"fall: {fall}, no_fall: {no_fall}")

    X = []
    y = []
    skipped = []
    paths = []
    label_map = {"Fall": 0,
                "No Fall": 1}

    for each_csv in tqdm(unique_files):
        try:
            df = pd.read_csv(each_csv)
            X.append(process_one_video(df, N=64))
            str_label = each_csv.parts[1]
            y.append(label_map[str_label])
            paths.append(str(each_csv)) # Keep track of the paths so we can do error analysis later

        # If there is a exception when running cell above, skipped and don't append it into the X and y list

        except Exception as e:
            skipped.append((each_csv, str(e)))

    print(f"preprocessed: {len(X)}, skipped: {len(skipped)}")

    if skipped:
        print(f"First three skipped files: {skipped[:3]}")

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(X, y, paths, train_size=0.8, shuffle=True, random_state=42)

    print(f"len of X_train: {len(X_train)}, len of X_test: {len(X_test)}, len of y_train: {len(y_train)}, len of y_test: {len(y_test)}")
    # Create train, test dataset
    train_data = FallDataset(X= X_train,
                             y= y_train,
                             transform= custom_transform,
                             augmentation= True)

    test_data = FallDataset(X= X_test,
                            y= y_test,
                            transform= custom_transform,
                            augmentation= False)

    # Create dataloader
    # BATCH_SIZE = 32
    # NUM_WORKERS = os.cpu_count() # Seems to be 12 when I run locally

    train_dataloader = DataLoader(dataset= train_data,
                                batch_size= BATCH_SIZE,
                                num_workers= NUM_WORKERS,
                                shuffle= True)

    test_dataloader = DataLoader(dataset= test_data,
                                batch_size= BATCH_SIZE,
                                num_workers= NUM_WORKERS,
                                shuffle= False) # must be false

    print(f"Len of train dataloader, should = number of batches: {len(train_dataloader)}")
    print(f"Len of test dataloader, should = number of batches: {len(test_dataloader)}")

    return train_dataloader, test_dataloader, paths_test


if __name__ == "__main__":
    train_dataloader, test_dataloader, paths_test = get_dataloaders()