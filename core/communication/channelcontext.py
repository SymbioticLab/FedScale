import logging
import grpc
import job_api_pb2_grpc
import job_api_pb2

class ExecutorConnections(object):
    """"Helps aggregator manage its grpc connection with executors."""
    MAX_MESSAGE_LENGTH = 50000000

    class _ExecutorContext(object):
        def __init__(self, executorId):
            self.id = executorId
            self.address = None
            self.channel = None
            self.stub = None

    def __init__(self, config, base_port=50000):
        self.executors = {}
        self.base_port = base_port

        executorId = 0
        for ip_numgpu in config.split("="):
            ip, numgpu = ip_numgpu.split(':')
            for numexe in numgpu.strip()[1:-1].split(','):
                for _ in range(int(numexe.strip())):
                    executorId += 1
                    self.executors[executorId] = ExecutorConnections._ExecutorContext(executorId)
                    self.executors[executorId].address = '{}:{}'.format(ip, self.base_port + executorId)

    def __len__(self):
        return len(self.executors)

    def __iter__(self):
        return iter(self.executors)

    def open_grpc_connection(self):
        for executorId in self.executors:
            logging.info('%%%%%%%%%% Opening grpc connection to ' + self.executors[executorId].address + ' %%%%%%%%%%')
            channel = grpc.insecure_channel(
                self.executors[executorId].address,
                options=[
                    ('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH),
                ]
            )
            self.executors[executorId].channel = channel
            self.executors[executorId].stub = job_api_pb2_grpc.JobServiceStub(channel)

    def close_grpc_connection(self):
        for executorId in self.executors:
            logging.info(f'%%%%%%%%%% Closing grpc connection with {executorId} %%%%%%%%%%')
            self.executors[executorId].channel.close()

    def get_stub(self, executorId):
        return self.executors[executorId].stub


