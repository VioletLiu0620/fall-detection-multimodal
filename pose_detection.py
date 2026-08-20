from ultralytics import YOLO
import numpy as np
import pandas as pd

model = YOLO("yolov8n-pose.pt")
results = model("c_f_s_a.MOV", stream= True) # stream= True reduce memory consumption
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
    df = df[["Frame", "Keypoint", "X", "Y", "Confidence" ]]

    result_list.append(df)

if result_list:
    final_df = pd.concat(result_list, ignore_index= True)
    final_df.to_csv("c_f_s_a.csv") # to_csv is a pd function that turn a Dataframe into a csv file

# for frame_idx, result in enumerate(results):
#     keypoints = result.keypoints.data # (num_ppl, 17, 3)
#     num_ppl = keypoints.shape[0]

#     if num_ppl == 0:
#         continue

#     for person in range(num_ppl):
#         kpt = keypoints[person] # (person1, 17, 3)
  
#         for i in range(kpt.shape[0]): # range of 1...17
#             x, y, conf = kpt[i]
#             result_list.append({
#                 "Frame": frame_idx,
#                 "Keypoint": joint_names[i],
#                 "X": x.item(),
#                 "Y": y.item(),
#                 "Confidence": conf.item()
#             })

