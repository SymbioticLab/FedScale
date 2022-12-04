# -*- coding: utf-8 -*-
import pickle

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.executor import Executor
from fedscale.cloud.execution.rlclient import RLClient
from fedscale.cloud.logger.execution import *
from fedscale.cloud import commons

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from async_client import Client as CustomizedClient

class AsyncExecutor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super().__init__(args)
        self.temp_model_path_version = lambda round: os.path.join(
            logDir, f'model_{self.this_rank}_{round}.pth.tar')

    def update_model_handler(self, model):
        """Update the model copy on this executor"""
        self.round += 1

        # Dump latest model to disk
        with open(self.temp_model_path_version(self.round), 'wb') as model_out:
            logging.info(
                f"Received latest model saved at {self.temp_model_path_version(self.round)}"
            )
            pickle.dump(model, model_out)

    def load_global_model(self, round=None):
        # load last global model
        # logging.info(f"====Load global model with version {round}")
        round = min(round, self.round) if round is not None else self.round
        with open(self.temp_model_path_version(round), 'rb') as model_in:
            model = pickle.load(model_in)
        return model

    def get_client_trainer(self, conf):
        return CustomizedClient(conf)

    def training_handler(self, clientId, conf, model=None):
        """Train model given client ids"""

        # Here model is model_id
        client_model = self.load_global_model(model)

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        if args.task == "rl":
            client_data = self.training_sets
            client = RLClient(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)
        else:
            client_data = select_dataset(clientId, self.training_sets,
                                         batch_size=conf.batch_size, args=self.args,
                                         collate_fn=self.collate_fn
                                         )

            client = self.get_client_trainer(conf)
            train_res = client.train(
                client_data=client_data, model=client_model, conf=conf)

        return train_res

    def testing_handler(self, args, config=None):

        evalStart = time.time()
        device = self.device
        model =   config['test_model']
        if self.task == 'rl':
            client = RLClient(args)
            test_res = client.test(args, self.this_rank, model, device=device)
            _, _, _, testResults = test_res
        else:
            data_loader = select_dataset(self.this_rank, self.testing_sets,
                                         batch_size=args.test_bsz, args=args,
                                         isTest=True, collate_fn=self.collate_fn
                                         )

            if self.task == 'voice':
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            if self.args.engine == commons.PYTORCH:
                test_res = test_model(self.this_rank, model, data_loader,
                                      device=device, criterion=criterion, tokenizer=tokenizer)
            else:
                raise Exception(f"Need customized implementation for model testing in {self.args.engine} engine")

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation round {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                         .format(self.round, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

    def check_model_version(self, model_id):
        return os.path.exists(self.temp_model_path_version(model_id))

    def remove_stale_models(self, straggler_round):
        """Remove useless models kept for async execution in the past"""
        logging.info(f"Current straggler round is {straggler_round}")
        stale_version = straggler_round-1
        while self.check_model_version(stale_version):
            logging.info(f"Executor {self.this_rank} removes stale model version {stale_version}")
            os.remove(self.temp_model_path_version(stale_version))
            stale_version -= 1

    def event_monitor(self):
        """Activate event handler once receiving new message
        """
        logging.info("Start monitoring events ...")
        self.client_register()

        while self.received_stop_request == False:
            if len(self.event_queue) > 0:
                request = self.event_queue.popleft()
                current_event = request.event

                logging.info(f"====Poping event {current_event}")
                if current_event == commons.CLIENT_TRAIN:
                    train_config = self.deserialize_response(request.meta)
                    train_model = self.deserialize_response(request.data)
                    if train_model is not None and not self.check_model_version(train_model):
                        # The executor may have not received the model due to async grpc
                        # TODO: server will lose track of scheduled but not executed task and remove the model
                        logging.error(f"Warning: Not receive model {train_model} for client {train_config['client_id'] }")
                        if self.round - train_model <= self.args.max_staleness:
                            self.event_queue.append(request)
                        time.sleep(1)
                        continue

                    train_config['model'] = train_model
                    train_config['client_id'] = int(train_config['client_id'])
                    client_id, train_res = self.Train(train_config)

                    # Upload model updates
                    future_call = self.aggregator_communicator.stub.CLIENT_EXECUTE_COMPLETION.future(
                        job_api_pb2.CompleteRequest(client_id=str(client_id), executor_id=self.executor_id,
                                                    event=commons.UPLOAD_MODEL, status=True, msg=None,
                                                    meta_result=None, data_result=self.serialize_response(train_res)
                                                    ))
                    future_call.add_done_callback(lambda _response: self.dispatch_worker_events(_response.result()))

                elif current_event == commons.MODEL_TEST:
                    test_configs = self.deserialize_response(request.meta)
                    self.remove_stale_models(test_configs['straggler_round'])
                    self.Test(test_configs)

                elif current_event == commons.UPDATE_MODEL:
                    broadcast_config = self.deserialize_response(request.data)
                    self.UpdateModel(broadcast_config)

                elif current_event == commons.SHUT_DOWN:
                    self.Stop()

                elif current_event == commons.DUMMY_EVENT:
                    pass
            else:
                time.sleep(1)
                self.client_ping()

if __name__ == "__main__":
    executor = AsyncExecutor(parser.args)
    executor.run()
