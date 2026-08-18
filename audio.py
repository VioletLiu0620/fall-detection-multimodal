import torch
import torchaudio
from torch_vggish_yamnet import vggish, yamnet
from torch_vggish_yamnet.input_proc import *
import pandas as pd

audio_model = yamnet.yamnet(pretrained=True)
audio_model.eval()

waveform, sample_rate = torchaudio.load("test_audio.wav")

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq= sample_rate,
                                            new_freq= 16000)
    waveform = resampler(waveform)
    sample_rate = 16000

if waveform.shape[0] > 1: # if channels is stero or more, not mono (channels, samples)
    waveform = waveform.mean(dim=0, keepdim=True)
    # Example: if it's stero, then mean average the left ear and right ear values, (2, 88200) -> (88200, ) but keepdim = True so (1, 88200)
    # keepdim = True keeps the dimension you are averaging on

# Input signal (x_in) tensor conversion & ad-hoc patching
converter = WaveformToInput()
in_tensor = converter(waveform=waveform.float(), sample_rate=sample_rate)
print(in_tensor.shape)

df = pd.read_csv("yamnet_class_map.csv")
print(df.head())
class_list = df["display_name"].tolist()
print(class_list[:5])

embed, scores = audio_model(in_tensor)     # (adjust order based on shapes)
print("embed shape:", embed.shape)  

avg_scores = scores.mean(dim=0)
print(avg_scores.shape)
print(avg_scores[:20])
top_score = avg_scores.argmax().item()
print(top_score)

print(f"The sound is most likely labeled as {class_list[top_score]}")