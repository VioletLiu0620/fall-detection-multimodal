import helper_functions
from helper_functions import Fall2d, mov_to_csv, csv_to_pred_label
from pathlib import Path
import torch

def predict(datafolder: str | Path,
            model_path: str | Path = "best_model_test_91acc.pth",
            device: str | torch.device = "cpu"):
    """
    Args: 
        datafolder (str | Path): the path where video files are located
        model_path (str | Path): state_dict model used, default to "best_model_test_91acc"
        device (str | torch.device): target device model runs on, default to cpu

    Returns:
        classification report
    """
    model = Fall2d(input_shape=3, output_shape= 2, hidden_units=10)
    model.to(device)

    datafolder_path = Path(datafolder)
    video_list = [video for video in datafolder_path.glob("*") if video.suffix.lower() in [".mp4", ".mov"]]

    for video_path in video_list:
        output_csv = mov_to_csv(video_path)
        pred_label = csv_to_pred_label(csv_file= output_csv, model= model, state_dict_path= model_path, device= device)

predict(datafolder= "custom_data")

