import threading

from fedscale.core import commons
from fedscale.core.resource_manager import ResourceManager as DefaultManager

class ResourceManager(DefaultManager):
    """Schedule training tasks across GPUs/CPUs"""

    def __init__(self, experiment_mode):
        super().__init__(experiment_mode)
        self.client_run_queue = []
        self.experiment_mode = experiment_mode
        self.update_lock = threading.Lock()

    def get_task_length(self):
        self.update_lock.acquire()
        remaining_task_num: int = len(self.client_run_queue)
        self.update_lock.release()
        return remaining_task_num

    def register_tasks(self, clientsToRun):
        self.client_run_queue += clientsToRun.copy()

    def has_next_task(self, client_id=None):
        exist_next_task = False
        if self.experiment_mode == commons.SIMULATION_MODE:
            exist_next_task = len(self.client_run_queue) > 0
        else:
            exist_next_task = client_id in self.client_run_queue
        return exist_next_task

    def get_next_task(self, client_id=None):
        next_task_id = None
        self.update_lock.acquire()
        if self.experiment_mode == commons.SIMULATION_MODE:
            if self.has_next_task(client_id):
                next_task_id = self.client_run_queue[0]
                self.client_run_queue.pop(0)
        else:
            if client_id in self.client_run_queue:
                next_task_id = client_id
                self.client_run_queue.remove(next_task_id)

        self.update_lock.release()
        return next_task_id
