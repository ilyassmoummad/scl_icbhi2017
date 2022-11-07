import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T

class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask=20, time_mask=50, freq_stripes=2, time_stripes=2, p=1.0):
        super().__init__()
        self.p = p
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.freq_stripes = freq_stripes
        self.time_stripes = time_stripes   
        self.specaugment = nn.Sequential(
            *[T.FrequencyMasking(freq_mask_param=self.freq_mask, iid_masks=True) for _ in range(self.freq_stripes)], 
            *[T.TimeMasking(time_mask_param=self.time_mask, iid_masks=True) for _ in range(self.time_stripes)],
            )
            
    def forward(self, audio):
        if self.p > torch.randn(1):
            return self.specaugment(audio)
        else:
            return audio