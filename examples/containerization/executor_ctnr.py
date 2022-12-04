"""An example of Executor container"""
import socket, json, time, logging
from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser

# We can do hard-code here because containers run in a separate virtual network
CONTAINER_IP = "0.0.0.0"
CONTAINER_PORT = 32000

class Executor_Wrapper(Executor):
    """Wrap Executor with a new init stage"""
    def __init__(self):
        new_args = self.wait_for_config()
        args_dict = vars(parser.args)
        for key in new_args:
            if key in args_dict:
                args_dict[key] = new_args[key]
            else:
                print(f'Executor_Wrapper: Warning: unrecognized argument {key} in config')
        super().__init__(parser.args)

    def wait_for_config(self):
        """ Initialization needed if executor is running in a container
        """
        # This is almost like aggregator's version execpt the logging, uniquely defined here to allow for future executor-specific init
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.bind((CONTAINER_IP, CONTAINER_PORT))
        listen_socket.settimeout(1)
        listen_socket.listen(5)
        print("Executor_Wrapper: Waiting to initialize")
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
                print("Executor_Wrapper: Error decoding init message!")
                listen_socket.close()
                exit(1)
            if msg['type'] == 'exec_init':
                print("Executor_Wrapper: Init success!")
                new_args = msg['data']
                # print(args)
                listen_socket.close()
                return new_args

if __name__ == "__main__":
    exec_ctnr = Executor_Wrapper()
    exec_ctnr.run()