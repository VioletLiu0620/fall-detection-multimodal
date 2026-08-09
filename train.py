import torch
import os
import numpy as np
import pandas as pd
from torch import nn
from data import get_dataloaders

train_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE=32, NUM_WORKERS=os.cpu_count())

# CNN 