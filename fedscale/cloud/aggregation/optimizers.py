import numpy as np
import torch


class TorchServerOptimizer(object):
    """This is a abstract server optimizer class

    Args:
        mode (string): mode of gradient aggregation policy
        args (distionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
        device (string): Runtime device type
        sample_seed (int): Random seed

    """

    def __init__(self, mode, args, device, sample_seed=233):
        self.mode = mode
        self.args = args
        self.device = device

        if mode == "fed-yogi":
            from fedscale.utils.optimizer.yogi import YoGi

            self.gradient_controller = YoGi(
                eta=args.yogi_eta,
                tau=args.yogi_tau,
                beta=args.yogi_beta,
                beta2=args.yogi_beta2,
            )

    def update_round_gradient(
        self, last_model, current_model, target_model, client_training_results=None
    ):
        """update global model based on different policy

        Args:
            last_model (list of tensor weight): A list of global model weight in last round.
            current_model (list of tensor weight): A list of global model weight in this round.
            target_model (PyTorch or TensorFlow nn module): Aggregated model.
            client_training_results list of gradients from every clients, for q-fedavg

        """
        if self.mode == "fed-yogi":
            """
            "Adaptive Federated Optimizations",
            Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan,
            ICLR 2021.
            """
            last_model = [x.to(device=self.device) for x in last_model]
            current_model = [x.to(device=self.device) for x in current_model]

            diff_weight = self.gradient_controller.update(
                [pb - pa for pa, pb in zip(last_model, current_model)]
            )

            new_state_dict = {
                name: torch.from_numpy(
                    np.array(last_model[idx] + diff_weight[idx], dtype=np.float32)
                )
                for idx, name in enumerate(target_model.state_dict().keys())
            }

            target_model.load_state_dict(new_state_dict)

        elif self.mode == "q-fedavg":
            """
            "Fair Resource Allocation in Federated Learning", Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith, ICLR 2020.
            """
            learning_rate, qfedq = self.args.learning_rate, self.args.qfed_q
            Deltas, hs = None, 0.0
            last_model = [x.to(device=self.device) for x in last_model]

            for result in client_training_results:
                # plug in the weight updates into the gradient
                update_weights = result["update_weight"]
                if type(update_weights) is dict:
                    update_weights = [x for x in update_weights.values()]

                weights = [
                    torch.tensor(x).to(device=self.device) for x in update_weights
                ]
                grads = [
                    (u - v) * 1.0 / learning_rate for u, v in zip(last_model, weights)
                ]
                loss = result["moving_loss"]

                if Deltas is None:
                    Deltas = [
                        np.float_power(loss + 1e-10, qfedq) * grad for grad in grads
                    ]
                else:
                    for idx in range(len(Deltas)):
                        Deltas[idx] += np.float_power(loss + 1e-10, qfedq) * grads[idx]

                # estimation of the local Lipchitz constant
                hs += qfedq * np.float_power(loss + 1e-10, (qfedq - 1)) * torch.sum(
                    torch.stack([torch.square(grad).sum() for grad in grads])
                ) + (1.0 / learning_rate) * np.float_power(loss + 1e-10, qfedq)

            # update global model
            new_state_dict = {
                name: last_model[idx] - Deltas[idx] / (hs + 1e-10) for idx, name in enumerate(target_model.state_dict().keys())
            }
            target_model.load_state_dict(new_state_dict)

        else:
            # The default optimizer, FedAvg, has been applied in aggregator.py on the fly
            pass
