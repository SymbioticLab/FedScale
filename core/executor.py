# -*- coding: utf-8 -*-
from fl_client_libs import *
from argparse import Namespace
import gc
from client import Client
from rlclient import RLClient
from concurrent import futures
import grpc
import job_api_pb2_grpc
import job_api_pb2
import io
import torch
import pickle


MAX_MESSAGE_LENGTH = 50000000


class Executor(job_api_pb2_grpc.JobServiceServicer):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
    def __init__(self, args):

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.num_executors = args.num_executors
        # ======== env information ========
        self.this_rank = args.this_rank

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ========
        self.server_event_queue = self.client_event_queue = None
        self.control_manager = None
        self.grpc_server = None

        # ======== runtime information ========
        self.collate_fn = None
        self.task = args.task
        self.epoch = 0
        self.start_run_time = time.time()
        self.received_stop_request = False

        super(Executor, self).__init__()


    def UpdateModel(self, request_iterator, context):
        """A GRPC functionfor JobService invoked by UpdateModel request.
        """
        logging.info('Received GRPC UpdateModel request')
        self.update_model_handler(request_iterator)
        return job_api_pb2.UpdateModelResponse()


    def Train(self, request, context):
        """A GRPC function for JobService invoked by Train request.
        """
        logging.info('Received GRPC Train request')
        return job_api_pb2.TrainResponse()


    def Stop(self, request, context):
        """A GRPC functionfor JobService invoked by Stop request.
        """
        logging.info('Received GRPC Stop request')
        self.received_stop_request = True
        return job_api_pb2.StopResponse()


    def ReportExecutorInfo(self, request, context):
        """A GRPC function for JobService invoked by ReportExecutorInfo request.

        This is called only once when the training starts.
        """
        logging.info('Received GRPC ReportExecutorInfo request')
        response = job_api_pb2.ReportExecutorInfoResponse()
        response.training_set_size.extend(self.training_sets.getSize()['size'])
        return response


    def Test(self, request, context):
        """A GRPC function for JobService invoked by Test request.
        """
        logging.info('Received GRPC Test request')
        test_res = self.testing_handler(args=self.args)
        return job_api_pb2.TestResponse(serialized_test_response=pickle.dumps(test_res))


    def setup_env(self):
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")

        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda:
            if self.device == None:
                for i in range(torch.cuda.device_count()):
                    try:
                        self.device = torch.device('cuda:'+str(i))
                        torch.cuda.set_device(i)
                        print(torch.rand(1).to(device=self.device))
                        logging.info(f'End up with cuda device ({self.device})')
                        break
                    except Exception as e:
                        assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'
            else:
                torch.cuda.set_device(self.device)

    def setup_communication(self):
        self.init_control_communication(self.args.ps_ip, self.args.manager_port)
        #self.init_data_communication()


    def setup_seed(self, seed=1):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self, ps_ip, ps_port):
        # Create communication channel between aggregator and worker
        # This channel serves control messages

        logging.info(f"Start to connect to {ps_ip}:{ps_port} for control plane communication ...")

        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=30),
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
        )
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, self.grpc_server)
        port = '[::]:{}'.format(50000 + self.this_rank)
        self.grpc_server.add_insecure_port(port)
        self.grpc_server.start()
        logging.info(f'Started GRPC server at {port}')

        BaseManager.register('get_server_event_que'+str(self.this_rank))
        BaseManager.register('get_client_event')

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        start_time, is_connected = time.time(), False

        while time.time() - start_time < 15 and not is_connected:
            try:
                self.control_manager.connect()
                is_connected = True
            except Exception as e:
                time.sleep(numpy.random.rand(1)[0]*5+0.1)
                logging.info(f"Retrying connection to {ps_ip}:{ps_port}")
                pass

        assert is_connected, 'Failed to connect to the aggregator'
        logging.info("Successfully connect to the aggregator")

        self.server_event_queue = eval('self.control_manager.get_server_event_que'+str(self.this_rank)+'()')
        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=self.num_executors + 1)


    def init_model(self):
        """Return the model architecture used in training"""
        return init_model()

    def init_data(self):
        """Return the training and testing dataset"""
        train_dataset, test_dataset = init_dataset()
        if self.task == "rl":
            return train_dataset, test_dataset
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(data=train_dataset, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(data=test_dataset, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=self.num_executors)

        logging.info("Data partitioner completes ...")


        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets


    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.model = self.model.to(device=self.device)
        self.training_sets, self.testing_sets = self.init_data()
        self.setup_communication()
        self.event_monitor()


    def push_msg_to_server(self, event, results):
        self.client_event_queue.put({'return': results, 'event': event, 'executorId': self.this_rank})


    def push_msg_to_server_asyn(self, event, results):
        self.client_event_queue.put_nowait({'return': results, 'event': event, 'executorId': self.this_rank})


    def report_executor_info_handler(self):
        """Return the statistics of training dataset"""
        return self.training_sets.getSize()


    def update_model_handler(self, request_iterator):
        """Update the model copy on this executor"""
        for param, request in zip(self.model.parameters(), request_iterator):
            buffer = io.BytesIO(request.serialized_tensor)
            buffer.seek(0)
            param.data = torch.load(buffer).to(device=self.device)

        self.epoch += 1
        if self.epoch % self.args.dump_epoch == 0 and self.this_rank == 1:
            with open(self.temp_model_path+'_'+str(self.epoch), 'wb') as model_out:
                pickle.dump(self.model, model_out)

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)


    def load_global_model(self):
        # load last global model
        try:
            with open(self.temp_model_path, 'rb') as model_in:
                model = pickle.load(model_in)
                return model
        except OSError as e:
            return self.model


    def override_conf(self, config):
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)


    def get_client_trainer(self, conf):
        """Developer can redefine to this function to customize the training:
           API:
            - train(client_data=client_data, model=client_model, conf=conf)
        """
        return Client(conf)


    def training_handler(self, clientId, conf):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model()

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size, collate_fn=self.collate_fn)

            client = self.get_client_trainer(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)

            # we need to get runtime variance for BN
            self.model = client_model
        return train_res


    def testing_handler(self, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, self.model, device=device)
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, isTest=True, collate_fn=self.collate_fn)

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            test_res = test_model(self.this_rank, self.model, data_loader, device=device, criterion=criterion, tokenizer=tokenizer)

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                        .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults


    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")

        while True:
            if self.received_stop_request:
                logging.info(f"Terminating (Executor {self.this_rank}) ...")
                self.grpc_server.stop(0)
                break

            if not self.server_event_queue.empty():
                event_dict = self.server_event_queue.get()
                event_msg = event_dict['event']

                logging.info(f"Executor {self.this_rank}: Received (Event:{event_msg.upper()}) from aggregator")

                # initiate each training round
                if event_msg == 'train':
                    clientId, client_conf = event_dict['clientId'], self.override_conf(event_dict['conf'])

                    train_res = self.training_handler(clientId=clientId, conf=client_conf)
                    self.push_msg_to_server('train_nowait', None)
                    # model updates may be time-consuming, thus we apply asyn push for better communication-computation overlaps
                    self.push_msg_to_server_asyn(event_msg, train_res)

                else:
                    logging.error("Unknown message types!")

            time.sleep(0.3)


    def stop(self):
        logging.info(f"Terminating (Executor {self.this_rank}) ...")


if __name__ == "__main__":
    executor = Executor(args)
    executor.run()

