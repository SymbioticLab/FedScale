import torch
from torch.nn import Module, Linear, Flatten, Softmax


class LinearModel(Module):
    """A simple linear model with Flatten->Linear.
    """
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear = Linear(3*28*28, 10)
        self.softmax = Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        return self.softmax(self.linear(self.flatten(input)))
