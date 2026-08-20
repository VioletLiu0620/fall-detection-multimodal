from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-pose.pt")
results = model("c_f_s_a.MOV", stream= True) # stream= True reduce memory consumption
result_list = []
joint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


for frame_idx, result in enumerate(results):
    keypoints = result.keypoints.data # (num_ppl, 17, 3)
    num_ppl = keypoints.shape[0]

    if num_ppl == 0:
        continue

    for person in range(num_ppl):
        kpt = keypoints[person] # (person1, 17, 3)
  
        for i in range(kpt.shape[0]): # range of 1...17
            x, y, conf = kpt[i]
            result_list.append({
                "Frame": frame_idx,
                "Keypoint": joint_names[i],
                "X": x.item(),
                "Y": y.item(),
                "Confidence": conf.item()
            })

