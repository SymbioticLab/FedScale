from fedscale.cloud.resource_manager import *


class AuxoResourceManager(ResourceManager):
    def __init__(self, experiment_mode):
        self.client_run_queue = [[]]
        self.client_run_queue_idx = [0]
        self.experiment_mode = experiment_mode
        self.update_lock = threading.Lock()


    def register_tasks(self, clientsToRun, cohort_id):
        self.client_run_queue[cohort_id] = clientsToRun.copy()
        self.client_run_queue_idx[cohort_id] = 0

    def split(self, cohort_id):
        self.client_run_queue.append( self.client_run_queue[cohort_id].copy())
        self.client_run_queue_idx.append(0)

    def get_task_length(self, cohort_id) -> int:
        """Number of tasks left in the queue

        Returns:
            int: Number of tasks left in the queue
        """
        self.update_lock.acquire()
        remaining_task_num: int = len(self.client_run_queue[cohort_id]) - self.client_run_queue_idx[cohort_id]
        self.update_lock.release()
        return remaining_task_num

    def remove_client_task(self, client_id, cohort_id):
        assert(client_id in self.client_run_queue[cohort_id],
               f"client task {client_id} is not in task queue")

    def has_next_task(self, client_id=None, cohort_id=0):
        exist_next_task = False
        if self.experiment_mode == commons.SIMULATION_MODE:
            exist_next_task = self.client_run_queue_idx[cohort_id] < len(
                self.client_run_queue[cohort_id])
        else:
            exist_next_task = client_id in self.client_run_queue[cohort_id]
        return exist_next_task

    def get_next_task(self, client_id=None, cohort_id=0):
        # TODO: remove client id when finish
        next_task_id = None
        self.update_lock.acquire()
        if self.experiment_mode == commons.SIMULATION_MODE:
            if self.has_next_task(client_id, cohort_id):
                next_task_id = self.client_run_queue[cohort_id][self.client_run_queue_idx[cohort_id]]
                self.client_run_queue_idx[cohort_id] += 1
        else:
            if client_id in self.client_run_queue[cohort_id]:
                next_task_id = client_id
                self.client_run_queue[cohort_id].remove(next_task_id)

        self.update_lock.release()
        return next_task_id