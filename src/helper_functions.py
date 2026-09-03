from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch
from torch import nn
import pathlib
from pathlib import Path
from preprocess import process_one_video

model = YOLO("yolov8n-pose.pt")

def mov_to_csv(video_path: str|Path):
    """
    Args:
        video_path (str | Path): full video path name, including the extension .mov or .mp4

    Returns:
        the name of a csv file with columns "Frame", "Keypoint", "X", "Y", "Confidence"
    """
    print(f"Processing video file {video_path} to csv")

    results = model(video_path, stream= True, verbose= False) # stream= True reduce memory consumption, verbose= False to stop printing long detection message

    result_list = []
    joint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]

    # Flatten version
    for frame_idx, result in enumerate(results):
        kpts = result.keypoints.data # (num_ppl, 17, 3)
        num_ppl = kpts.shape[0] # DON'T DO kpts[0].shape -> this will error and kpts[0] gives you the first object in the array

        if num_ppl == 0:
            continue

        # FIX: If multiple people/artifacts are detected, keep only the first track
        if num_ppl > 1:
            kpts = kpts[0:1] # Slice to keep only shape (1, 17, 3)
            num_ppl = 1

        flat_kpts = kpts.reshape(-1, 3).cpu().numpy() # from (num_ppl, 17, 3) but now treat it as (num_ppl*17, 3), flatten to 2D from a 3D matrix
        df = pd.DataFrame(flat_kpts, columns= ["X", "Y", "Confidence"])

        df["Frame"] = frame_idx + 1
        df["Keypoint"] = np.tile(joint_names, num_ppl) # repeat (joint_names) for (num_ppl) times

        # Rearrange
        df = df[["Frame", "Keypoint", "X", "Y", "Confidence"]]

        result_list.append(df)

    video_path = str(video_path)
    video_name = video_path.split(".")[0]
    output_csv_name = f"{video_name}.csv"

    if result_list:
        final_df = pd.concat(result_list, ignore_index= True)
        final_csv = final_df.to_csv(output_csv_name, index=False) # to_csv is a pd function that turn a Dataframe into a csv file

    print(f"Outputing {output_csv_name}")

    return output_csv_name


def csv_to_pred_label(csv_file: str | Path,
                      model: nn.Module,
                      fall_folder_name: str,
                      nofall_folder_name: str,
                      device: str | torch.device = "cpu" ):
    """
    Args:
        csv_file (str | Path): str or path of the csv file
        model (nn.Module): any PyTorch model for classification
        fall_folder_name (str): the directory name of where fall videos are located
        nofall_folder_name (str): the directory name of where no fall/ADL videos are located
        device (str | Path): device that the model be on, default to cpu

    Returns:
        pred_label (str): the label that model predicted on - "Fall" or "No Fall" (labels are the f/nf folder name that user given)
    """
    label_map = {0: fall_folder_name,
                 1: nofall_folder_name}
    
    arr = process_one_video(video_csv= pd.read_csv(csv_file))
    X = torch.from_numpy(arr).float().unsqueeze(dim=0).to(device)

    model.to(device)

    model.eval()
    with torch.inference_mode():
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_label = label_map[y_pred.item()]

    # file_path = Path(csv_file) # doesn't matter whether csv_file is a string or path, filename becomes a path object
    # filename = file_path.stem

    # print(f"The file {filename} is classified as {y_label}")

    return y_label

class Fall2d(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels= input_shape, 
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride= 1,
                      padding= 0), # (64, 17) -> (62, 15)
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,
                      out_channels= hidden_units,
                      kernel_size= 3,
                      stride= 1,
                      padding= 0), # (62, 15) -> (60, 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride= 2), # MaxPool stride's default is the same as the kernel size
            # (60, 13) -> (30, 6)

            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(out_features= output_shape)
        )
    def forward(self, x):
        # Originally we have torch.Size([32, 64, 17, 3]): (batches, frames, joints, coords), but coords is what we want to see the difference across frames and joints
        # Therefore, we need to permute it because of PyTorch convention that (batches, channels, height, width) -> (32, 3, 64, 17)

        return self.conv_layers(x.permute(0, 3, 1, 2))

