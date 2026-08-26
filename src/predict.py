from helper_functions import Fall2d, mov_to_csv, csv_to_pred_label
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def predict(datafolder: str | Path,
            fall_folder_name: str,
            nofall_folder_name: str,
            model_path: str | Path = "best_model_test_91acc.pth",
            device: str | torch.device = "cpu"):
    """
    Args: 
        datafolder (str | Path): the path where video files are located
        fall_folder_name (str): the directory name of where fall videos are located
        nofall_folder_name (str): the directory name of where no fall/ADL videos are located
        model_path (str | Path): state_dict model used, default to "best_model_test_91acc"
        device (str | torch.device): target device model runs on, default to cpu

    Returns:
        classification report
    """
    model = Fall2d(input_shape=3,
                   output_shape= 2,
                   hidden_units=16)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    datafolder_path = Path(datafolder)
    video_list = []
    y_trues = []

    for video in datafolder_path.rglob("*"):
        if video.suffix.lower() in [".mp4", ".mov"]:
            video_list.append(video)
            y_trues.append(video.parent.name)
    
    print(f"Total number of videos: {len(video_list)}")
    print("Unique true labels found:", set(y_trues))
    y_preds = []

    for video_path in tqdm(video_list):
        output_csv = mov_to_csv(video_path)
        pred_label = csv_to_pred_label(csv_file= output_csv,
                                       model= model,
                                       fall_folder_name= fall_folder_name,
                                       nofall_folder_name= nofall_folder_name,
                                       device= device)
        y_preds.append(pred_label)

    display_labels = [fall_folder_name, nofall_folder_name]

    confmat = confusion_matrix(y_true= y_trues,
                               y_pred= y_preds,
                               labels= display_labels)
    
    display = ConfusionMatrixDisplay(confusion_matrix= confmat,
                                     display_labels= display_labels)
    
    display.plot()
    plt.show()

    report = classification_report(y_true= y_trues,
                                   y_pred= y_preds,
                                   labels= display_labels,
                                   target_names= display_labels)
    
    print(report)
    

predict(datafolder= "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos",
        fall_folder_name= "Fall",
        nofall_folder_name= "ADL")

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
