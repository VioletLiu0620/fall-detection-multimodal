import torch
import os
import numpy as np
import pandas as pd
from torch import nn
from data import get_dataloaders
from helper_functions import Fall2d
from timeit import default_timer as timer
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

train_dataloader, test_dataloader, paths_test = get_dataloaders(BATCH_SIZE=16, NUM_WORKERS=0) # could be os_count() but use 0 here for mac's child process issue

# NOTE: Fall2d is defined once, in helper_functions.py, and imported here so
# training and inference are always guaranteed to use the exact same architecture.

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

    best_test_loss = float("inf")
    
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

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best test loss at epoch: {epoch}, test_loss: {test_loss}")

    return result


model = Fall2d(input_shape=3,
               output_shape=2,
               hidden_units=16)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                            lr= 0.001)

start_time = timer()

result = train(model= model,
               train_dataloader= train_dataloader,
               test_dataloader= test_dataloader,
               loss_fn= loss_fn,
               optimizer= optimizer,
               epochs= 30,
               device= device)

end_time = timer()

print(f"Total training time {end_time - start_time:.3f} seconds")

# Plot the confusion matrix

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_preds = []
y_trues = []
wrong_files = []
idx = 0

with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
        y_trues.append(y.cpu())

        for i in range(len(y)):
            if y_pred[i] != y[i]:
                wrong_files.append({
                    "files": paths_test[idx],
                    "pred": y_pred[i].item(),
                    "true": y[i].item()
                })
            idx += 1

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

print(f"Number of False Detection: {len(wrong_files)}")
false_positive = [w for w in wrong_files if w["pred"] == 0 and w["true"] == 1] # false alarm, fall = 0, no fall = 1
false_negative = [w for w in wrong_files if w["pred"] == 1 and w["true"] == 0] # missed alarm, more serious

print(f"Number of false positive (false alarm): {len(false_positive)}")
for w in false_positive:
    print(f"{w['files']}")

print(f"Number of false negative (missed alarm): {len(false_negative)}")
for w in false_negative:
    print(f"{w['files']}")


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