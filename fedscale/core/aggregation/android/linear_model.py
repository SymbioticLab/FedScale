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
        """Linear model forward

        Args:
            input (torch.Tensor): A tensor input with shape (N, 3, 28, 28).

        Returns:
            torch.Tensor: A tensor output with shape (N, 10), 
                          which is the classification prediction of the model.
        """
        return self.softmax(self.linear(self.flatten(input)))
