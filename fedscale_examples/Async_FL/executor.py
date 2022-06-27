# -*- coding: utf-8 -*-
from fedscale.core.fl_client_libs import *
from argparse import Namespace
import gc
import collections
import torch
import pickle

from fedscale.core.executor import Executor
from fedscale.core.client import Client
from fedscale.core.rlclient import RLClient
from fedscale.core import events
from fedscale.core.communication.channel_context import ClientConnections
import fedscale.core.job_api_pb2 as job_api_pb2


class AsyncExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
    def __init__(self, args):
        Executor.__init__(self, args)
        self.temp_model_path_version = lambda round: os.path.join(logDir, 'model_' + str(round) + '.pth.tar')

    def Train(self, config):
        """Load train config and data to start training on client """
        client_id, train_config = config['client_id'], config['task_config']

        model = None
        if 'model' in train_config and train_config['model'] is not None:
            model = train_config['model']

        client_conf = self.override_conf(train_config)
        train_res = self.training_handler(clientId=client_id, conf=client_conf, model=model)

        # Report execution completion meta information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id = str(client_id), executor_id = self.executor_id,
                event = events.CLIENT_TRAIN, status = True, msg = None,
                meta_result = None, data_result = None
            )
        )
        self.dispatch_worker_events(response)

        return client_id, train_res

    def Test(self, config):
        """Model Testing. By default, we test the accuracy on all data of clients in the test group"""

        test_res = self.testing_handler(args=self.args)
        test_res = {'executorId': self.this_rank, 'results': test_res}

        # Report execution completion information
        response = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION(
            job_api_pb2.CompleteRequest(
                client_id = self.executor_id, executor_id = self.executor_id,
                event = events.MODEL_TEST, status = True, msg = None,
                meta_result = None, data_result = self.serialize_response(test_res)
            )
        )
        self.dispatch_worker_events(response)

    def update_model_handler(self, model):
        """Update the model copy on this executor"""
        self.model = model
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path_version(self.round), 'wb') as model_out:
            logging.info(f"Received latest model saved at {self.temp_model_path_version(self.round)}")
            pickle.dump(self.model, model_out)

    def load_global_model(self, round):
        # load last global model
        if round == -1:
            with open(self.temp_model_path, 'rb') as model_in:
                model = pickle.load(model_in)
        else:
            round = min(round, self.round)
            with open(self.temp_model_path_version(round), 'rb') as model_in:
                model = pickle.load(model_in)
        return model

    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model(-1) if model is None \
                                    else self.load_global_model(model)

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                batch_size=conf.batch_size, args = self.args,
                collate_fn=self.collate_fn
            )

            client = self.get_client_trainer(conf)
            train_res = client.train(client_data=client_data, model=client_model, conf=conf)

        return train_res


    def testing_handler(self, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        model = self.load_global_model(self.round)
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets,
                batch_size=args.test_bsz, args = self.args,
                isTest=True, collate_fn=self.collate_fn
            )

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            if self.args.engine == events.PYTORCH:
                test_res = test_model(self.this_rank, model, data_loader,
                    device=device, criterion=criterion, tokenizer=tokenizer)
            else:
                raise Exception(f"Need customized implementation for model testing in {self.args.engine} engine")

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                .format(self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                if current_event == events.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    _ = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id = str(client_id), executor_id = self.executor_id,
                        event = events.UPLOAD_MODEL, status = True, msg = None,
                        meta_result = None, data_result = self.serialize_response(train_res)
                    ))

                elif current_event == events.MODEL_TEST:
                    self.Test(self.deserialize_response(request.meta))

                elif current_event == events.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)

                elif current_event == events.SHUT_DOWN:
                    self.Stop()

                elif current_event == events.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()


if __name__ == "__main__":
    executor = AsyncExecutor(args)
    executor.run()