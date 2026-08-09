from preprocess import process_one_video
import torch
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

data_path = Path("data")
all_keypoints_list = list(data_path.glob("**/*.csv"))

print(f"Total numbers of video: {len(all_keypoints_list)}")

fall = sum(1 for p in all_keypoints_list if "No Fall" not in str(p))
no_fall = sum(1 for p in all_keypoints_list if "No Fall" in str(p))
print(f"fall: {fall}, no_fall: {no_fall}")

X = []
y = []
skipped = []
label_map = {"Fall": 0,
             "No Fall": 1}

for each_csv in tqdm(all_keypoints_list):
    try:
        df = pd.read_csv(each_csv)
        X.append(process_one_video(df, N=64))
        str_label = each_csv.parts[1]
        y.append(label_map[str_label])
    # If there is a exception when running cell above, skipped and don't append it into the X and y list

    except Exception as e:
        skipped.append((each_csv, str(e)))

print(f"preprocessed: {len(X)}, skipped: {len(skipped)}")

if skipped:
    print(f"First three skipped files: {skipped[:3]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

print(f"len of X_train: {len(X_train)}, len of X_test: {len(X_test)}, len of y_train: {len(y_train)}, len of y_test: {len(y_test)}")

## We need to normalize: Center on hip
def custom_transform(video_array):
    ar = video_array.copy()
    xy = ar[:, :, :2]
    print(xy.shape)
    conf = ar[:, :, 2:]

    LEFT_HIP, RIGHT_HIP = 11, 12
    mid_hip = (xy[:, LEFT_HIP, :] + xy[:, RIGHT_HIP, :]) / 2 # midhip.shape = (64, 2)
    xy =  xy - mid_hip[:, None, :] # change midhip into (64, 1, 2) to be able minus xy
    result = np.concatenate((xy, conf), axis=2) # conf is joining at last dimenstion, xy: (64, 17, 2) + conf: (64, 17, 1) - > result: (64, 17, 3)

    return result

# first = X_train[0]
# print(f"Shape before normalize: {first.shape}")
# result = custom_transfrom(first)
# print(result)
# print(f"Shape after normalize: {result.shape}")

## Create custom dataset class
class FallDataset(Dataset):
    def __init__(self, X, y, transform= None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        label = self.y[index]

        if self.transform:
            x = self.transform(x)

        # convert to tensor
        x = torch.tensor(x, dtype= torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return x, label

# Create train, test dataset
train_data = FallDataset(X= X_train,
                         y= y_train,
                         transform= custom_transform)

test_data = FallDataset(X= X_test,
                        y= y_test,
                        transform= custom_transform)

# Create dataloader
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() # Seems to be 12 when I run locally

train_dataloader = DataLoader(dataset= train_data,
                              batch_size= BATCH_SIZE,
                              num_workers= NUM_WORKERS,
                              shuffle= True)

test_dataloader = DataLoader(dataset= test_data,
                             batch_size= BATCH_SIZE,
                             num_workers= NUM_WORKERS,
                             shuffle= False)

print(f"Len of train dataloader, should = number of batches: {len(train_dataloader)}")
print(f"Len of test dataloader, should = number of batches: {len(test_dataloader)}")