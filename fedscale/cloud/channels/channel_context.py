import logging

import grpc

import fedscale.cloud.channels.job_api_pb2_grpc as job_api_pb2_grpc

MAX_MESSAGE_LENGTH = 1*1024*1024*1024  # 1GB


class ClientConnections(object):
    """"Clients build connections to the cloud aggregator."""

    def __init__(self, aggregator_address, base_port=18888):
        self.base_port = base_port
        self.aggregator_address = aggregator_address
        self.channel = None
        self.stub = None

    def connect_to_server(self):
        logging.info('%%%%%%%%%% Opening grpc connection to ' +
                     self.aggregator_address + ' %%%%%%%%%%')
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.aggregator_address, self.base_port),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ]
        )
        self.stub = job_api_pb2_grpc.JobServiceStub(self.channel)

    def close_sever_connection(self):
        logging.info(
            '%%%%%%%%%% Closing grpc connection to the aggregator %%%%%%%%%%')
        self.channel.close()
