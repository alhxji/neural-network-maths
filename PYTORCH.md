# Why PYTORCH! What's different.

Pytorch simply makes life easier and buildig models less insufferable. It replaces a lot of our mathematical operation and provides abstractions that performs the same function behind the scene. If you think of the maths version as writing just raw SQl, then pytorch is like an ORM/ Framework.

It replaces a lot of thing, a few examples are:

### 1. The Model itself
```
model = nn.Sequential(
    nn.Linear(784, 10),
    nn.ReLU(),
    nn.Linear(10, 10)
)
```
As crazy as it might sound, that's our entire network in pytorch.

### 2. nn.Linear(784,10)
Replaces the whole first layer, the weight, the bias, does almost everything for us.

### 2. ReLU
Instead of the raw ReLU function we wrote, there is a function `torch.nn.ReLU()`

And a bunch of other things that makes writing models easier, you should check it out

# Code
[pytorch.py](./pytorch.py)