import torch

class YoGi():
    def __init__(self, eta=1e-2, tau=1e-3, beta=0.999, beta2=-1):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.v_t = []
        self.m_t = []
        self.beta2 = beta2

    def update(self, gradients):
        update_gradients = []
        for idx, gradient in enumerate(gradients):
            gradient_square = gradient * gradient
            if len(self.v_t) <= idx:
                self.v_t.append(gradient_square)
                self.m_t.append(gradient)
            else:
                # yogi
                self.v_t[idx] = self.v_t[idx] - (1.-self.beta) * gradient_square * torch.sign(self.v_t[idx] - gradient_square)
                yogi_learning_rate = self.eta /(torch.sqrt(self.v_t[idx]) + self.tau)

                if self.beta2 != -1:
                    # adam
                    self.m_t[idx] = self.beta2 * self.m_t[idx] + (1.-self.beta2) * gradient

                if self.beta2 == -1:
                    update_gradients.append(yogi_learning_rate * gradient)
                else:
                    # add the momentum
                    update_gradients.append(yogi_learning_rate * self.m_t[idx])


        if len(update_gradients) == 0:
            update_gradients = gradients

        return update_gradients
