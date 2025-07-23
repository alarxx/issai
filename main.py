import os
import math

import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tkinter
# tkinter._test()
# print(matplotlib.get_backend())

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F
import torch.optim as optim

# utils
from utils.data.CustomImageDataset import CustomImageDataset
from utils.devices import print_available_devices
print_available_devices()
from utils.plot import plot4
from utils.save import save

# ---

if __name__ == "__main__":

    CHANNELS, HEIGHT, WIDTH = 3, 28, 28

    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),  # например
        transforms.ToTensor(), # tensor(C, H, W) in RGB
    ])

    dataset = CustomImageDataset(root_dir="dataset/CMNIST", transform=transform)

    # Как разделить на Train, Val, Test
    # validation_data = K-Fold Cross Validation on training_data
    training_data, test_data = dataset.train_test_split(test_size=0.25, random_state=42)

    # ---

    BATCH_SIZE = 256

    # Create data loaders.
    # Data Loader wraps an iterable over dataset
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for X, y in test_dataloader:
    # X, y = next(iter(test_dataloader))
        print(f"Shape of X [N, C, H, W]: {X.shape}") # torch.Size([64, 1, 28, 28]) torch.float32 [0; 1]
        print(f"Shape of y: {y.shape} {y.dtype}") # torch.Size([64]) torch.int64
        break

    # ---

    def visualize(dataset, classes):
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]
            print(img.shape)
            figure.add_subplot(rows, cols, i)
            plt.title(classes[label])
            plt.axis("off")
            plt.imshow(img.permute(1, 2, 0)) # tensor(C, H, W), а метод принимает img(H, W, C)
        plt.show()

    visualize(dataset, dataset.classes)

    # ---

    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")

    # ---

    # Define model
    class MLP(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.flatten = nn.Flatten() # flattens an image to row-vector [<->], so rows are samples -> shape[n, 784]
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(CHANNELS * HEIGHT * WIDTH, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, len(classes))
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

        def __setattr__(self, name, value):
            print(f"__setattr__ called: {name} = {value}")
            super().__setattr__(name, value)

    model = MLP(classes=dataset.classes).to(device)
    # print(type(nn.Sequential()))
    # print(type(nn.Module()))
    # print(type(model))
    # print(model)
    # print(model.parameters())
    # for param in model.parameters():
    #     print(param) # tensors
    print(f"Model structure: {model}\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values[:2] : {param[:2]} \n")

    # ---

    with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
        X = torch.rand(2, CHANNELS, HEIGHT, WIDTH, device=device) # (N, C, H, W)
        model.eval()
        logits = model(X)
        print(logits.shape)
        pred_probab = nn.Softmax(dim=1)(logits) # along row, columns are entries
        print(pred_probab)
        y_pred = pred_probab.argmax(dim=1) # along row
        # y_pred = [y1, y2]
        print(f"Predicted class: {y_pred}")

    # ---

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # *math.sqrt(BATCH_SIZE) # doesn't work with AdamW

    def train(dataloader, model, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.train()
        train_loss, correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum()
            # print("loss:", loss.dtype, "; correct:", correct.dtype)

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * dataloader.batch_size # len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # skip training
            if batch == 5:
                break
        train_loss /= num_batches
        correct /= size
        return train_loss.item(), correct.item()


    def test(dataloader, model, loss_fn, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
        with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum()
                # skip training
                if batch == 5:
                    break
            test_loss /= num_batches
            correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss.item(), correct.item()


    EPOCHS = 5
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        # train
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        # test
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        # save
        save(model,
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            test_losses=test_losses,
            test_accuracies=test_accuracies,
            filename="checkpoint",
            index=t
        )
    print("Done!")

    # ---

    # save(model,
    #      train_losses=train_losses,
    #      train_accuracies=train_accuracies,
    #      test_losses=test_losses,
    #      test_accuracies=test_accuracies,
    #      filename="checkpoint",
    # )

    # ---

    model = MLP(classes=dataset.classes).to(device)

    checkpoint = torch.load(f"checkpoint_{EPOCHS-1}.pth", weights_only=False)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    plot4(
        checkpoint["train_losses"],
        checkpoint["train_accuracies"],
        checkpoint["test_losses"],
        checkpoint["test_accuracies"]
    )

    print(f"Test loaded model\n-------------------------------")
    test(test_dataloader, model, loss_fn, device)

    model.eval()
    x, y = test_data[0]
    with torch.no_grad():
        x = x.unsqueeze(0).to(device) # [N, C, H, W]
        pred = model(x)
        predicted, actual = dataset.classes[pred[0].argmax(0).item()], dataset.classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
