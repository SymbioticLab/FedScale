"""A skeleton for Pytorch Executor"""
from fedscale.cloud.execution.executor import Executor



class Torch_Executor(Executor):
    """A class for PyTorch version of Executor, directly inherited from fedscale/cloud/execution/executor.py"""
    pass

if __name__ == "__main__":
    torch_exec = Torch_Executor()
    torch_exec.run()