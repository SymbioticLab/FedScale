import logging

class ResourceManager(object):
    """Schedule training tasks across GPUs/CPUs"""

    def __init__(self):

        self.client_run_queue = []
        self.client_run_queue_idx = 0

    
    def register_tasks(self, clientsToRun):
        self.client_run_queue = clientsToRun
        self.client_run_queue_idx = 0


    def get_next_task(self):
        if self.client_run_queue_idx < len(self.client_run_queue):
            clientId = self.client_run_queue[self.client_run_queue_idx]
            self.client_run_queue_idx += 1
            return clientId

        return None

