import torch
import numpy as np


class YoGi:
    def __init__(self, eta=1e-2, tau=1e-3, beta=0.9, beta2=0.99):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.v_t = []
        self.m_t = []
        self.beta2 = beta2

    def update(self, gradients):
        update_gradients = []
        if not self.v_t:
            self.v_t = [torch.full_like(g, self.tau) for g in gradients]
            self.m_t = [torch.full_like(g, 0.0) for g in gradients]

        for idx, gradient in enumerate(gradients):
            gradient_square = gradient**2

            self.m_t[idx] = self.beta * self.m_t[idx] + (1.0 - self.beta) * gradient

            self.v_t[idx] = self.v_t[idx] - (
                1.0 - self.beta2
            ) * gradient_square * torch.sign(self.v_t[idx] - gradient_square)
            yogi_learning_rate = self.eta / (torch.sqrt(self.v_t[idx]) + self.tau)

            update_gradients.append(yogi_learning_rate * self.m_t[idx])

        if len(update_gradients) == 0:
            update_gradients = gradients

        return update_gradients
