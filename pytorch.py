# Implementation of what we already have in main.py but in pytorch
# Pytorch doesn;t change the mathematics, it just provides abstractions that does the job automatically

# LFG!

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


data = pd.read_csv("data/train.csv").values

np.random.shuffle(data)

data_test  = data[:1000]
data_train = data[1000:]


# split into training and testing data

y_test = torch.tensor(
    data_test[:, 0],
    dtype=torch.long
)

X_test = torch.tensor(
    data_test[:,1:] / 255.0,
    dtype=torch.float32
)

y_train = torch.tensor(
    data_train[:, 0],
    dtype=torch.long
)

X_train = torch.tensor(
    data_train[:,1:] / 255.0,
    dtype=torch.float32
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# Dataset and dataloader

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# THE MODEL!

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 10)
        self.relu =  nn.ReLU()
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

# that's all. just standard practice and oop

model = NeuralNetwork()


# now for gradient loss and backward prop(loss and optimizer here)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1 # learning rate
)


# the training

epochs = 10 # 1 epoch mean one run through the whole dataset

for epoch in range(epochs):

    model.train() # training mode

    for X, y in train_loader:

        # forward prop
        output = model(X)

        # The loss
        loss = loss_fn(output, y)

        # Backwary prop
        optimizer.zero_grad() # empty the gradient so it does;t compound
        loss.backward()

        # update the params
        optimizer.step()

    print(
        f"epoch {epoch + 1}/{epochs}"
        f"Loss: {loss.item():.4f}"
    )


# Testing

model.eval() # evalution mode

correct = 0
total = 0

with torch.no_grad(): # no autograd

    for X, y in test_loader:

        output = model(X)

        predictions = output.argmax(dim=1)

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

accuracy = correct / total

print(f"Test accuracy: {accuracy * 100:.2f}%")

# 91.8% accuracy on first run! LFG!