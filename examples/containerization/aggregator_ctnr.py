"""An example of containerizing Aggregator"""
import socket, json, time, logging
from fedscale.cloud.aggregation.aggregator import Aggregator
import fedscale.cloud.config_parser as parser

# We can do hard-code here because containers run in a separate virtual network
CONTAINER_IP = "0.0.0.0"
CONTAINER_PORT = 30000

class Aggregator_Wrapper(Aggregator):
    """Wrap Aggregator with a new init stage"""
    def __init__(self):
        new_args = self.wait_for_config()
        args_dict = vars(parser.args)
        for key in new_args:
            if key in args_dict:
                args_dict[key] = new_args[key]
            else:
                print(f'Aggregator_Wrapper: Warning: unrecognized argument {key} in config')
        super().__init__(parser.args)
        
    def wait_for_config(self):
        """ Initialization needed if aggregator is running in a container
        """
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.bind((CONTAINER_IP, CONTAINER_PORT))
        listen_socket.settimeout(1)
        listen_socket.listen(5)
        print("Aggregator_Wrapper: Waiting to initialize")
        while True:
            # avoid busy waiting
            time.sleep(0.1)
            try:
                # wait for connection
                incoming_socket, addr = listen_socket.accept()
            except socket.timeout:
                continue
            message_chunks = []
            while True:
                time.sleep(0.1)
                try:
                    # receive messages
                    msg = incoming_socket.recv(4096)
                except socket.timeout:
                    continue
                if not msg:
                    break
                message_chunks.append(msg)
            message_bytes = b''.join(message_chunks)
            # decode messages
            message_str = message_bytes.decode('utf-8')
            incoming_socket.close()
            try:
                msg = json.loads(message_str)
            except json.JSONDecodeError:
                print("Aggregator_Wrapper: Error decoding init message!")
                listen_socket.close()
                exit(1)
            if msg['type'] == 'aggr_init':
                print("Aggregator_Wrapper: Init success!")
                new_args = msg['data']
                # print(args)
                listen_socket.close()
                return new_args

if __name__ == "__main__":
    aggr_ctnr = Aggregator_Wrapper()
    aggr_ctnr.run()