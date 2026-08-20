from preprocess import process_one_video
import torch
from torch import nn
import pandas as pd
import numpy as np

csv_path = "c_f_s_a.csv"
video_df = pd.read_csv(csv_path)
X = torch.from_numpy(process_one_video(video_csv=video_df)).float().unsqueeze(dim=0)


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

label_map = {"Fall": 0,
             "No Fall": 1}

class_names = list(label_map.keys())
 
model = Fall2d(input_shape=3,
               output_shape=2,
               hidden_units=16)

model.load_state_dict(torch.load("best_model_test_91acc.pth"))

model.eval()

with torch.inference_mode():
    y_logits = model(X)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1).item()
    y_label = class_names[y_pred]

print(f"The model classified {csv_path} as {y_label}")

