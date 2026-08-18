import torch
import torchaudio
from torch_vggish_yamnet import vggish, yamnet
from torch_vggish_yamnet.input_proc import *

audio_model = yamnet.yamnet(pretrained=True)
audio_model.eval()

waveform, sample_rate = torchaudio.load("test_audio.wav")

print(waveform.shape)
print(sample_rate)

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq= sample_rate,
                                            new_freq= 16000)
    waveform = resampler(waveform)

if waveform.shape[0] > 1: # if channels is stero or more, not mono (channels, samples)
    waveform = waveform.mean(dim=0, keepdim=True)
    # Example: if it's stero, then mean average the left ear and right ear values, (2, 88200) -> (88200, ) but keepdim = True so (1, 88200)
    # keepdim = True keeps the dimension you are averaging on

print(waveform.shape)