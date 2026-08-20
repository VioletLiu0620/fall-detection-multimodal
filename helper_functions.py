from ultralytics import YOLO
import numpy as np
import pandas as pd

model = YOLO("yolov8n-pose.pt")

def mov_to_csv(video_path: str):
    """
    Parameters:
    video_name: name of the video file without .mov or .mp4
    video_type: .mov or .mp4, default to .mov 

    Returns a csv file that has the same name as video_name.csv with columns "Frame", "Keypoint", "X", "Y", "Confidence"
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

        flat_kpts = kpts.reshape(-1, 3).cpu().numpy() # from (num_ppl, 17, 3) but now treat it as (num_ppl*17, 3), flatten to 2D from a 3D matrix
        df = pd.DataFrame(flat_kpts, columns= ["X", "Y", "Confidence"])

        df["Frame"] = frame_idx + 1
        df["Keypoint"] = np.tile(joint_names, num_ppl) # repeat (joint_names) for (num_ppl) times

        # Rearrange
        df = df[["Frame", "Keypoint", "X", "Y", "Confidence"]]

        result_list.append(df)

    video_name = video_path.split(".")[0]
    output_csv_name = f"{video_name}.csv"

    if result_list:
        final_df = pd.concat(result_list, ignore_index= True)
        final_csv = final_df.to_csv(output_csv_name, index=False) # to_csv is a pd function that turn a Dataframe into a csv file

    print(f"Outputing {output_csv_name}")

    return final_csv

mov_to_csv("c_f_c_a.MOV")