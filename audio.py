import torch
import torchaudio
from torch_vggish_yamnet import vggish, yamnet
from torch_vggish_yamnet.input_proc import *
import pandas as pd
from pathlib import Path

audio_model = yamnet.yamnet(pretrained=True)
audio_model.eval()

audio_folder = Path("audio_data")
audio_list = audio_folder.glob("*.wav")

df = pd.read_csv("yamnet_class_map.csv")
class_list = df["display_name"].tolist()

# the distress classes we care about for falls
distress_keywords = ["scream", "shout", "yell", "groan", "thud", "thump", "crying"]
distress_indices = [i for i, name in enumerate(class_list)
                    if any(k in name.lower() for k in distress_keywords)]

for audio_file in audio_list:

    waveform, sample_rate = torchaudio.load(audio_file)

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
    # print(in_tensor.shape)

    embed, scores = audio_model(in_tensor)     # (adjust order based on shapes)
    # print("embed shape:", embed.shape)  
    
    # MAX aggregation across frames (peak per class) -> good for brief sounds like a fall
    max_scores = scores.max(dim=0)[0]          # (521,)
    max_probs = torch.sigmoid(max_scores)      # convert logits -> 0-1 probabilities
 
    print(f"=== {audio_file.name} ===")
 
    # top 5 overall (by peak probability)
    top5 = max_probs.topk(5)
    print("  top 5:")
    for score, idx in zip(top5.values, top5.indices):
        print(f"    {class_list[idx.item()]}: {score.item():.3f}")
 
    # distress classes, using MAX-based probs (the fix: was printing avg before)
    print("  distress (max):")
    for i in distress_indices:
        print(f"    {class_list[i]}: {max_probs[i].item():.3f}")