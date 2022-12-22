from typing import List

import numpy as np
import torch

from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.model_wrapper_base import ModelWrapperBase


class TorchModelWrapper(ModelWrapperBase):
    def __init__(self, model: torch.nn.Module, optimizer: TorchServerOptimizer = None):
        self.model = model
        self.optimizer = optimizer

    def set_weights(self, weights: List[np.ndarray]):
        current_grad_weights = [param.data.clone() for param in self.model.state_dict().values()]
        new_state_dict = {
            name: torch.from_numpy(np.asarray(weights[i])) for i, name in enumerate(self.model.state_dict().keys())
        }
        self.model.load_state_dict(new_state_dict)
        if self.optimizer:
            self.optimizer.update_round_gradient(weights, current_grad_weights, self.model)

    def get_weights(self) -> List[np.ndarray]:
        return [params.data.clone() for params in self.model.state_dict().values()]

    def get_model(self):
        return self.model
