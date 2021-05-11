# -*- coding: utf-8 -*-
from fl_client_libs import *
from fl_client_libs import tokenizer, collate, voice_collate_fn, args
from argparse import Namespace

# initiate the log path, and executor ips
initiate_client_setting()
os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
os.environ['GLOO_SOCKET_IFNAME'] = 'enP5p7s0f1'


class Executor(object):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
    def __init__(self, args):
        
        self.args = args
        self.device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
        self.executors = [int(v) for v in str(args.learners).split('-')]

        # ======== env information ======== 
        self.this_rank = args.this_rank

        # ======== model and data ======== 
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ======== 
        self.server_event_queue = self.client_event_queue = None
        self.control_manager = None

        # ======== runtime information ======== 
        self.collate_fn = None
        self.task = args.task
        self.epoch = 0
        self.start_run_time = time.time()


    def setup_env(self):
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")

        self.setup_seed(seed=self.this_rank)
        
        # set up device
        if self.args.use_cuda:
            for i in range(torch.cuda.device_count()):
                try:
                    self.device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    print(torch.rand(1).to(device=self.device))
                    logging.info(f'End up with cuda device ({self.device})')
                    break
                except Exception as e:
                    assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'

        self.init_control_communication(self.args.ps_ip, self.args.manager_port)
        self.init_data_communication()


    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self, ps_ip, ps_port):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Start to connect to {ps_ip}:{ps_port} for control plane communication ...")

        #for executor_id in self.executors:
        BaseManager.register('get_server_event_que'+str(self.this_rank))
        BaseManager.register('get_client_event')

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        self.control_manager.connect()

        self.server_event_queue = eval('self.control_manager.get_server_event_que'+str(self.this_rank)+'()')
        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)


    def init_model(self):
        return init_model()

    def init_data(self):
        """Load data and """
        train_dataset, test_dataset = init_dataset()
        
        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(data=train_dataset, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(data=test_dataset, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=len(self.executors))

        logging.info("Data partitioner completes ...")


        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets

    def run_client(self, client_data, model, conf):
        args = conf

        clientId = args.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        device = self.device

        model = model.to(device=device)
        trained_unique_samples = min(len(client_data), args.local_steps) * args.batch_size
        
        global_model = [param.data.clone() for param in model.parameters()]
        if args.task == "detection":
            lr = args.learning_rate
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        elif args.task == 'nlp':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)


        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=5e-4)
        criterion = CTCLoss(reduction='none').to(device=device) if args.task=='voice' else torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

        epoch_train_loss = 1e-4#None
        count = 0
        model.train()

        error_type = None

        completed_steps = 0
        if args.task == "detection":
            im_data = Variable(torch.FloatTensor(1).cuda())
            im_info = Variable(torch.FloatTensor(1).cuda())
            num_boxes = Variable(torch.LongTensor(1).cuda())
            gt_boxes = Variable(torch.FloatTensor(1).cuda())
        # TODO: if indeed enforce FedAvg, we will run fixed number of epochs, instead of iterations

        while completed_steps < args.local_steps:
            try:
                for data_pair in client_data:

                    if args.task == 'nlp':
                        (data, _) = data_pair
                        data, target = mask_tokens(data, tokenizer, args, device=device) if args.mlm else (data, data)
                    elif args.task == 'voice':
                        (data, target, input_percentages, target_sizes), _ = data_pair
                        input_sizes = input_percentages.mul_(int(data.size(3))).int()
                    elif args.task == 'detection':
                        temp_data = data_pair
                        target = temp_data[4]
                        data = temp_data[0:4]
                    else:
                        (data, target) = data_pair

                    if args.task == "detection":
                        im_data.resize_(data[0].size()).copy_(data[0])
                        im_info.resize_(data[1].size()).copy_(data[1])
                        gt_boxes.resize_(data[2].size()).copy_(data[2])
                        num_boxes.resize_(data[3].size()).copy_(data[3])
                    elif args.task == 'speech':
                        data = torch.unsqueeze(data, 1).to(device=device)
                    else:
                        data = Variable(data).to(device=device)

                    target = Variable(target).to(device=device)

                    if args.task == 'nlp':
                        outputs = model(data, masked_lm_labels=target) if args.mlm else model(data, labels=target)
                        loss = outputs[0]
                    elif args.task == 'voice':
                        outputs, output_sizes = model(data, input_sizes)
                        outputs = outputs.transpose(0, 1).float()  # TxNxH
                        loss = criterion(outputs, target, output_sizes, target_sizes)
                    elif args.task == "detection":
                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label = model(im_data, im_info, gt_boxes, num_boxes)
                    
                        loss = rpn_loss_cls + rpn_loss_box \
                                + RCNN_loss_cls + RCNN_loss_bbox

                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    else:
                        output = model(data)
                        loss = criterion(output, target)

                    # ======== collect training feedback for other decision components [e.g., kuiper selector] ======
                    if args.task == 'nlp':
                        loss_list = [loss.item()]
                    elif args.task == "detection":
                        loss_list = [loss.tolist()]
                    else:
                        loss_list = loss.tolist()

                    temp_loss = sum([l**2 for l in loss_list])/float(len(loss_list))

                    # only measure the loss of the first epoch
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - args.loss_decay) * epoch_train_loss + args.loss_decay * temp_loss

                    # ========= Define the backward loss ==============
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    # ========= Weight handler ========================
                    if args.proxy_avg:
                        for idx, param in enumerate(model.parameters()):
                            param.data += args.learning_rate * args.proxy_mu * (param.data - global_model[idx])

                    count += len(target)
                    completed_steps += 1

                    if completed_steps == args.local_steps:
                        break

            except Exception as ex:
                error_type = ex
                break

        model_param = [param.data.cpu().numpy() for param in model.parameters()]
        results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss, 
                  'trained_size': count, 'wall_duration': 0, 'success': count > 0, 
                  'utility':math.sqrt(epoch_train_loss)*trained_unique_samples}

        if count > 0:
            logging.info(f"Training of (CLIENT: {clientId}) completes")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        return results


    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.model = self.model.to(device=self.device)
        self.training_sets, self.testing_sets = self.init_data()
        self.event_monitor()


    def push_msg_to_server(self, event, results):
        self.client_event_queue.put({'return': results, 'event': event, 'executorId': self.this_rank})


    def push_msg_to_server_asyn(self, event, results):
        self.client_event_queue.put_nowait({'return': results, 'event': event, 'executorId': self.this_rank})


    def report_executor_info_handler(self):
        return self.training_sets.getSize()


    def update_model_handler(self):
        self.epoch += 1

        """Update the model copy on this executor"""
        for param in self.model.parameters():
            temp_tensor = torch.zeros_like(param.data, device='cpu')
            dist.recv(tensor=temp_tensor, src=0)
            param.data = temp_tensor.to(device=self.device)

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)


    def load_global_model(self):
        # load last global model
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model


    def override_conf(self, config):
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)

    def training_handler(self, clientId, conf):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model()

        conf.clientId = clientId
        client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size, collate_fn=self.collate_fn)
        train_res = self.run_client(client_data=client_data, model=client_model, conf=conf)

        # we need to get runtime variance for BN
        self.model = client_model
        return train_res


    def testing_handler(self, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, isTest=True, collate_fn=self.collate_fn)
        criterion = CTCLoss(reduction='mean').to(device=device) if self.task == 'voice' else torch.nn.CrossEntropyLoss().to(device=device)

        test_res = test_model(self.this_rank, self.model, data_loader, device=device, criterion=criterion, tokenizer=tokenizer)

        test_loss, acc, acc_5, testResults = test_res
        logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                    .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        return testResults


    def event_monitor(self):
        logging.info("Start monitoring events ...")

        while True:
            if not self.server_event_queue.empty():
                event_dict = self.server_event_queue.get()
                event_msg = event_dict['event']

                logging.info(f"Received (Event:{event_msg.upper()}) from aggregator")

                if event_msg == 'report_executor_info':
                    executor_info = self.report_executor_info_handler()
                    self.push_msg_to_server(event_msg, executor_info)

                elif event_msg == 'update_model':
                    self.update_model_handler()

                # initiate each training round
                elif event_msg == 'train':
                    clientId, client_conf = event_dict['clientId'], self.override_conf(event_dict['conf'])

                    train_res = self.training_handler(clientId=clientId, conf=client_conf)
                    self.push_msg_to_server('train_nowait', None)
                    # model updates may be time-consuming, thus we apply asyn push for better communication-computation overlaps
                    self.push_msg_to_server_asyn(event_msg, train_res)

                elif event_msg == 'test':
                    test_res = self.testing_handler(args=self.args)
                    self.push_msg_to_server(event_msg, test_res)

                elif event_msg == 'stop':
                    self.stop()
                    break

                else:
                    logging.error("Unknown message types!")

                time.sleep(0.1)


    def stop(self):
        logging.info(f"Terminating (Executor {self.this_rank}) ...")

        self.control_manager.shutdown()


if __name__ == "__main__":
    executor = Executor(args)
    executor.run()

