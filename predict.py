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
    model = Fall2d(input_shape=3, output_shape= 2, hidden_units=16)
    model.to(device)

    datafolder_path = Path(datafolder)
    video_list = [video for video in datafolder_path.glob("*") if video.suffix.lower() in [".mp4", ".mov"]]

    fall_label_count = 0

    for video_path in video_list:
        output_csv = mov_to_csv(video_path)
        pred_label = csv_to_pred_label(csv_file= output_csv, model= model, state_dict_path= model_path, device= device)
        if pred_label == "Fall":
            fall_label_count += 1
        print(f"\n------------------\n")

    print(f"Total number of videos: {len(video_list)}")
    print(f"Total number of video labeled as Fall: {fall_label_count}")

predict(datafolder= "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/Subject 4/ADL")

### Dataset: GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
# Citation:
#       E. Alam, A. Sufian, P. Dutta, M. Leo, I. A. Hameed "GMDCSA24: A Dataset for Human Fall Detection in Videos", Data in Brief (communicated).

# ADL stands for Activities of Daily Living

# Subject 1 Results:
# Fall: 10/16 
# ADL: 16/16 

# Subject 2 Results:
# Fall: 15/25
# ADL: 20/23

# Subject 3 Results:
# Fall: 15/21
# ADL: 20/22

# Subject 4 Results:
# Fall: 12/17
# ADL: 16/20
