import logging

from fedscale.cloud.internal.client_metadata import *

class AuxoClientMetadata(ClientMetadata):
    def __init__(self, host_id, client_id, speed, traces=None):
        super().__init__(host_id, client_id, speed, traces)
        self.grad_ratio = 10

    def register_gradient(self, W, W_old):
        W_old = [dt.cpu().numpy() for dt in W_old]
        W = [W[k] for k in W]
        gradient = [pb - pa for pa, pb in zip(W, W_old)]
        gradient = np.concatenate([v.flatten() for v in gradient])
        val_len = len(gradient) // self.grad_ratio  # change grad size
        self.gradient = np.float16(gradient[-val_len:])
        return self.gradient
