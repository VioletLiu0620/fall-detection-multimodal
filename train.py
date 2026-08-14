import torch
import os
import numpy as np
import pandas as pd
from torch import nn
from data import get_dataloaders
from timeit import default_timer as timer
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

train_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE=32, NUM_WORKERS=0) # could be os_count() but use 0 here for mac's child process issue

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
            
            nn.Flatten(),
            nn.LazyLinear(out_features= output_shape)
        )
    def forward(self, x):
        # Originally we have torch.Size([32, 64, 17, 3]): (batches, frames, joints, coords), but coords is what we want to see the difference across frames and joints
        # Therefore, we need to permute it because of PyTorch convention that (batches, channels, height, width) -> (32, 3, 64, 17)

        return self.conv_layers(x.permute(0, 3, 1, 2))

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):

    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_logit = model(X)

        loss = loss_fn(y_logit, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_label = torch.softmax(y_logit, dim=1).argmax(dim=1)
        train_acc += (y_label == y).sum().item() / len(y_label)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device = device):

    model.to(device)
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_logit = model(X)

            loss = loss_fn(y_logit, y)
            test_loss += loss.item()

            y_label = torch.softmax(y_logit, dim=1).argmax(dim=1)
            test_acc += (y_label == y).sum().item() / len(y_label)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device = device):

    model.to(device)
    
    result = {"train loss": [],
              "train acc": [],
              "test loss": [],
              "test acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model= model,
                                           dataloader= train_dataloader,
                                           loss_fn= loss_fn,
                                           optimizer= optimizer,
                                           device= device)
        
        test_loss, test_acc = test_step(model= model,
                                        dataloader= test_dataloader,
                                        loss_fn= loss_fn,
                                        device= device)

        print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

        result["train loss"].append(train_loss)
        result["train acc"].append(train_acc)
        result["test loss"].append(test_loss)
        result["test acc"].append(test_acc)

    return result


model = Fall2d(input_shape=3,
               output_shape=2,
               hidden_units=32)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                            lr= 0.001)

start_time = timer()

result = train(model= model,
               train_dataloader= train_dataloader,
               test_dataloader= test_dataloader,
               loss_fn= loss_fn,
               optimizer= optimizer,
               epochs= 5,
               device= device)

end_time = timer()

print(f"Total training time {end_time - start_time:.3f} seconds")

# Plot the confusion matrix

model.eval()
y_preds = []
y_trues = []
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
        y_trues.append(y.cpu())

y_preds = torch.cat(y_preds)     # combine 34 batch-tensors into one (1086,) tensor
y_trues = torch.cat(y_trues)     # same for the true labels

confmat = confusion_matrix(y_true= y_trues,
                           y_pred= y_preds)

display = ConfusionMatrixDisplay(confusion_matrix= confmat,
                                 display_labels=["Fall", "No Fall"])

display.plot()
plt.show()

report = classification_report(y_true= y_trues,
                               y_pred= y_preds,
                               target_names=["Fall", "No Fall"])

print(report)

## Test for model actually working

# sample, label = next(iter(train_dataloader))
# model = Fall2d(input_shape= 3,
#                output_shape= 2,
#                hidden_units= 32)
# dummy = model(sample)
# print(dummy)
# print(dummy.shape)


## Test for train_dataloader actually working and size of batches

# print(sample)
# print(index) # tensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1])
# print(sample.shape) # torch.Size([32, 64, 17, 3])
# print(index.shape) # torch.Size([32]) the label