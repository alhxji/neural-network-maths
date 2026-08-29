# Just jargons and testinng!

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

print(torch.__version__)
print(torch.backends.mps.is_available())

x = torch.tensor([1,2,3,4])

# print(x)
# print(x.shape)
# print(x.dtype)

data = pd.read_csv("data/train.csv")

# print(data.head())
# print(data.shape)

data = np.array(data)
np.random.shuffle(data)

data_test = data[:1000].T
data_train = data[1000:].T

label_test = data_test[0]
feature_test = data_test[1:] / 255.0

label_train = data_train[0]
feature_train = data_train[1:] / 255.0

# transpose because pytroch uses a differnte model, uses number of images X features
X_train = torch.tensor(feature_train.T, dtype = torch.float32)
y_train = torch.tensor(label_train, dtype = torch.long)

X_test = torch.tensor(feature_test.T, dtype = torch.float32)
y_test = torch.tensor(label_test, dtype = torch.long)

# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

train_dataset = TensorDataset(
    X_train, y_train
)

test_dataset = TensorDataset(
    X_test, y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)

for X, y in train_loader:
    print(X.shape)
    print(y.shape)
    break



layer = nn.Linear(784, 10)

# print(layer.weight.shape)
# print(layer.bias.shape)

# nn.ReLU() # quite handy if you ask me

# nn.CrossEntropyLoss() # expects the unnormalized logits from the output of nn.Linear()

# model = nn.Sequential(
#     nn.Linear(784, 10),
#     nn.ReLU(),
#     nn.Linear(10, 10)
# )

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

# z = x ** 3

y.backward()
# z.backward()

# print(x.grad)


# loss_fn = nn.CrossEntropyLoss()

# output = model(X_train)

# loss = loss_fn(output, y_train)

# loss.backward()
