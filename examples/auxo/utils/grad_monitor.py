import logging
import numpy as np
import math

class Gradient_Monitor():
    def __init__(self, rank):
        self.client_grad = {}
        self.client_data = {}
        self.grad_stability = []
        self.rank = rank


    def register_client(self, client_id, new_w, old_w, client_data):
        if self.rank > 1:
            return

        old_w = [dt.cpu().numpy() for dt in old_w]
        new_w = [new_w[k] for k in new_w]
        gradient = [pb - pa for pa, pb in zip(new_w, old_w)]
        self.client_grad[client_id] = np.concatenate([n.ravel() for n in gradient])
        clt_data = None
        for data_pair in client_data:
            (data, target) = data_pair
            if clt_data is None:
                clt_data = [np.asarray(data.ravel())]
            else:
                clt_data.append(np.asarray(data.ravel()))

        self.client_data[client_id] = np.mean(clt_data, axis=0)
        logging.info(f'client_data[{client_id}] registered')

    def _cal_data_similarity(self, client_1, client_2):
        '''Calculate the cosine similarity between the data of two clients'''
        data_1 = self.client_data[client_1].ravel()
        data_2 = self.client_data[client_2].ravel()
        cosine_similarity = np.dot(data_1, data_2) / (np.linalg.norm(data_1) * np.linalg.norm(data_2))
        return cosine_similarity

    def _cal_grad_similarity(self, client_1, client_2):
        '''Calculate the cosine similarity between the gradient of two clients'''
        grad_1 = self.client_grad[client_1]
        grad_2 = self.client_grad[client_2]

        # Calculate the magnitude (Euclidean norm) of A and B
        magnitude_A = np.linalg.norm(grad_1)
        magnitude_B = np.linalg.norm(grad_2)
        # Calculate the cosine similarity
        cosine_similarity = np.dot(grad_1, grad_2) / (magnitude_A * magnitude_B)

        return cosine_similarity

    def cal_pairwise_grad_stability(self):
        '''Calculate the pairwise gradient similarity and data similarity'''
        if len(self.client_grad) > 0:
            grad_sim = []
            data_sim = []
            for client_id1 in self.client_grad:
                for client_id2 in self.client_grad:
                    if client_id1 > client_id2:
                        data_similarity = self._cal_data_similarity(client_id1, client_id2)
                        grad_similarity = self._cal_grad_similarity(client_id1, client_id2)
                        # logging.info(f"Gradient similarity between {client_id1} and {client_id2}: {grad_similarity}")
                        # logging.info(f"Data similarity between {client_id1} and {client_id2}: {data_similarity}")
                        data_sim.append(data_similarity)
                        grad_sim.append(grad_similarity)
            correlation_coefficient = np.corrcoef(grad_sim, data_sim)[0, 1]

            self.grad_stability.append(correlation_coefficient)
            logging.info(f"Gradient stability: {self.grad_stability}")
            # Reset for the next round
            self.client_grad = {}
            self.client_data = {}

