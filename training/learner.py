# -*- coding: utf-8 -*-
from fl_client_libs import *

initiate_client_setting()

for i in range(torch.cuda.device_count()):
    try:
        device = torch.device('cuda:'+str(i))
        torch.cuda.set_device(i)
        logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
        break
    except Exception as e:
        assert i == torch.cuda.device_count()-1, 'Can not find a feasible GPU'

world_size = 0
global_trainDB = None
global_testDB = None
last_model_tensors = []
nextClientIds = None
global_data_iter = {}
global_client_profile = {}
global_optimizers = {}
sampledClientSet = set()

# for malicious experiments only
malicious_clients = set()
flip_label_mapping = {}

workers = [int(v) for v in str(args.learners).split('-')]

os.environ['MASTER_ADDR'] = args.ps_ip
os.environ['MASTER_PORT'] = args.ps_port
# os.environ['NCCL_DEBUG'] = 'INFO'

logging.info("===== Experiment start =====")

# =================== Label flipper ================ #
def generate_flip_mapping(num_of_labels, random_seed=0):
    global flip_label_mapping

    from random import Random

    rng = Random()
    rng.seed(random_seed)

    label_mapping = list(range(num_of_labels))
    rng.shuffle(label_mapping)

    flip_label_mapping = {x: label_mapping[x] for x in range(num_of_labels)}

    logging.info("====Flip label mapping is: \n{}".format(flip_label_mapping))

def generate_malicious_clients(compromised_ratio, num_of_clients, random_seed=0):
    global malicious_clients

    from random import Random

    rng = Random()
    rng.seed(random_seed)

    shuffled_client_ids = list(range(num_of_clients))
    rng.shuffle(shuffled_client_ids)

    trunc_len = int(compromised_ratio * num_of_clients)
    malicious_clients = set(shuffled_client_ids[:trunc_len])

    logging.info("====Malicious clients are: \n{}".format(malicious_clients))

# =================== Report client information ================ #
def report_data_info(rank, queue):
    global nextClientIds, global_trainDB

    client_div = global_trainDB.getDistance()
    # report data information to the clientSampler master
    queue.put({
        rank: [client_div, global_trainDB.getSize()]
    })

    clientIdToRun = torch.zeros([world_size - 1], dtype=torch.int).to(device=device)
    dist.broadcast(tensor=clientIdToRun, src=0)
    nextClientIds = [clientIdToRun[args.this_rank - 1].item()]

    if args.malicious_clients > 0:
        generate_malicious_clients(args.malicious_clients, len(client_div))
        generate_flip_mapping(args.num_class)

def init_myprocesses(rank, size, model,
                   q, param_q, stop_flag,
                   fn, backend, client_cfg):
    print("====Worker: init_myprocesses")
    fn(rank, model, q, param_q, stop_flag, client_cfg)

def scan_models(path):
    files = os.listdir(path)
    model_paths = {}

    for file in files:
        if not os.path.isdir(file):
            if '.pth.tar' in file and args.model in file:
                model_state_id = int(re.findall(args.model+"_(.+?).pth.tar", file)[0])
                model_paths[model_state_id] = os.path.join(path, file)

    return model_paths

# ================== Scorer =================== #

def collate(examples):
    global tokenizer

    if tokenizer._pad_token is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), None)

def voice_collate_fn(batch):
    def func(p):
        return p[0].size(1)

    start_time = time.time()

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)

    end_time = time.time()

    return (inputs, targets, input_percentages, target_sizes), None

# =================== simulating different clients =====================#

def run_client(clientId, cmodel, iters, learning_rate, argdicts = {}):
    global global_trainDB, global_data_iter, last_model_tensors, tokenizer
    global malicious_clients, flip_label_mapping

    logging.info(f"Start to run client {clientId} ...")

    curBatch = -1

    if args.task != 'nlp' and args.task != 'text_clf':
        optimizer = MySGD(cmodel.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in cmodel.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 5e-4,
            },
            {"params": [p for n, p in cmodel.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon, weight_decay=5e-4)

    criterion = None

    if args.task == 'voice':
        criterion = CTCLoss(reduction='none').to(device=device)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

    train_data_itr_list = []
    collate_fn = None

    if args.task == 'nlp':
        collate_fn = collate
    elif args.task == 'voice':
        collate_fn = voice_collate_fn

    if clientId not in global_data_iter:
        client_train_data = select_dataset(
                                clientId, global_trainDB,
                                batch_size=args.batch_size,
                                collate_fn=collate_fn
                            )

        train_data_itr = iter(client_train_data)
        total_batch_size = len(train_data_itr)
        global_data_iter[clientId] = [train_data_itr, curBatch, total_batch_size, argdicts['iters']]
    else:
        [train_data_itr, curBatch, total_batch_size, epo] = global_data_iter[clientId]

    local_trained = 0
    epoch_train_loss = None
    comp_duration = 0.
    norm_gradient = 0.
    count = 0

    train_data_itr_list.append(train_data_itr)
    run_start = time.time()

    numOfFailures = 0
    numOfTries = 5

    input_sizes, target_sizes, output_sizes = None, None, None
    masks = None
    is_malicious = True if clientId in malicious_clients else False

    cmodel = cmodel.to(device=device)
    cmodel.train()

    # TODO: if indeed enforce FedAvg, we will run fixed number of epochs, instead of iterations
    for itr in range(iters):
        it_start = time.time()
        fetchSuccess = False

        while not fetchSuccess and numOfFailures < numOfTries:
            try:
                try:
                    if args.task == 'nlp':
                        # target is None in this case
                        (data, _) = next(train_data_itr_list[0])
                        data, target = mask_tokens(data, tokenizer, args) if args.mlm else (data, data)
                    elif args.task == 'text_clf':
                        (data, masks), target = next(train_data_itr_list[0])
                        masks = Variable(masks).to(device=device)
                    elif args.task == 'voice':
                        (data, target, input_percentages, target_sizes), _ = next(train_data_itr_list[0])
                        input_sizes = input_percentages.mul_(int(data.size(3))).int()
                    else:
                        (data, target) = next(train_data_itr_list[0])

                    fetchSuccess = True
                except Exception as ex:
                    try:
                        if args.num_loaders > 0:
                            train_data_itr_list[0]._shutdown_workers()
                            del train_data_itr_list[0]
                    except Exception as e:
                        logging.info("====Error {}".format(str(e)))

                    tempData = select_dataset(
                            clientId, global_trainDB,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn
                        )

                    #logging.info(f"====Error {str(ex)}")
                    train_data_itr_list = [iter(tempData)]

            except Exception as e:
                numOfFailures += 1
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
                time.sleep(0.5)

        if numOfFailures >= numOfTries:
            break

        numOfFailures = 0
        curBatch = curBatch + 1

        # flip the label if the client is malicious
        if is_malicious:
            for idx, x in enumerate(target):
                target[idx] = flip_label_mapping[int(x.item())]

        data = Variable(data).to(device=device)
        if args.task != 'voice':
            target = Variable(target).to(device=device)
        if args.task == 'speech':
            data = torch.unsqueeze(data, 1)

        local_trained += len(target)

        comp_start = time.time()

        if args.task == 'nlp':
            outputs = cmodel(data, masked_lm_labels=target) if args.mlm else cmodel(data, labels=target)
            loss = outputs[0]
            #torch.nn.utils.clip_grad_norm_(cmodel.parameters(), args.max_grad_norm)
        elif args.task == 'text_clf':
            loss, logits = cmodel(data, token_type_ids=None, attention_mask=masks, labels=target)
            #loss = criterion(output, target)
        elif args.task == 'voice':
            outputs, output_sizes = cmodel(data, input_sizes)
            outputs = outputs.transpose(0, 1).float()  # TxNxH
            loss = criterion(outputs, target, output_sizes, target_sizes).to(device=device)
        else:
            output = cmodel(data)
            loss = criterion(output, target)

        temp_loss = 0.
        loss_cnt = 1.

        loss_list = loss.tolist() if args.task != 'nlp' else [loss.item()]
        for l in loss_list:
            temp_loss += l**2

        loss_cnt = len(loss_list)

        temp_loss = temp_loss/float(loss_cnt)

        # only measure the loss of the first epoch
        if itr < total_batch_size:# >= (iters - total_batch_size - 1):
            if epoch_train_loss is None:
                epoch_train_loss = temp_loss
            else:
                epoch_train_loss = (1. - args.loss_decay) * epoch_train_loss + args.loss_decay * temp_loss

        count += len(target)

        # ========= Define the backward loss ==============
        optimizer.zero_grad()
        loss.mean().backward()

        if args.task != 'nlp' and args.task != 'text_clf':
            delta_w = optimizer.get_delta_w(learning_rate)

            if not args.proxy_avg:
                for idx, param in enumerate(cmodel.parameters()):
                    param.data -= delta_w[idx].to(device=device)
            else:
                for idx, param in enumerate(cmodel.parameters()):
                    param.data -= delta_w[idx].to(device=device)
                    param.data += learning_rate * args.proxy_mu * (last_model_tensors[idx] - param.data)
        else:
            # proxy term
            optimizer.step()

            if args.proxy_avg:
                for idx, param in enumerate(cmodel.parameters()):
                    param.data += learning_rate * args.proxy_mu * (last_model_tensors[idx] - param.data)

            cmodel.zero_grad()

        comp_duration = (time.time() - comp_start)

        #logging.info('For client {}, upload iter {}, epoch {}, Batch {}/{}, Loss:{} | TotalTime: {} | CompTime: {} | DataLoader: {} | epoch_train_loss: {} | malicious: {}\n'
         #           .format(clientId, argdicts['iters'], int(curBatch/total_batch_size),
         #           (curBatch % total_batch_size), total_batch_size, temp_loss,
         #           round(time.time() - it_start, 4), round(comp_duration, 4), round(comp_start - it_start, 4), epoch_train_loss, is_malicious))

    # remove the one with LRU
    if len(global_client_profile) > args.max_iter_store:
        allClients = global_data_iter.keys()
        rmClient = sorted(allClients, key=lambda k:global_data_iter[k][3])[0]

        del global_data_iter[rmClient]

    # save the state of this client if # of batches > iters, since we want to pass over all samples at least one time
    if total_batch_size > iters * 10 and len(train_data_itr_list) > 0 and not args.release_cache:
        global_data_iter[clientId] = [train_data_itr_list[0], curBatch, total_batch_size, argdicts['iters']]
    else:
        if args.num_loaders > 0:
            for loader in train_data_itr_list:
                try:
                    loader._shutdown_workers()
                except Exception as e:
                    pass
        del train_data_itr_list
        del global_data_iter[clientId]

    # we only transfer the delta_weight
    model_param = [(param.data - last_model_tensors[idx]).cpu().numpy() for idx, param in enumerate(cmodel.parameters())]

    time_spent = time.time() - run_start

    # add bias to the virtual clock, computation x (# of trained samples) + communication
    if clientId in global_client_profile:
        time_cost = global_client_profile[clientId][0] * count + global_client_profile[clientId][1]
    else:
        time_cost = time_spent

    speed = 0
    isSuccess = True
    if count > 0:
        speed = time_spent/float(count)
    else:
        isSuccess = False
        logging.info("====Failed to run client {}".format(clientId))

    logging.info(f"Completed to run client {clientId}")

    #logging.info("====Epoch epoch_train_loss is {}".format(epoch_train_loss))
    return model_param, epoch_train_loss, local_trained, str(speed) + '_' + str(count), time_cost, isSuccess

def run(rank, model, queue, param_q, stop_flag, client_cfg):
    logging.info("====Worker: Start running")

    global nextClientIds, global_trainDB, global_testDB, last_model_tensors
    criterion = None

    if args.task == 'voice':
        criterion = CTCLoss(reduction='mean').to(device=device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device=device)

    startTime = time.time()

    # Fetch the initial parameters from the server side (we called it parameter_server)
    last_model_tensors = []
    #model.train()
    model = model.to(device=device)

    for idx, param in enumerate(model.parameters()):
        dist.broadcast(tensor=param.data, src=0)

    if args.load_model:
        try:
            with open(modelPath, 'rb') as fin:
                model = pickle.load(fin)

            model = model.to(device=device)
            #model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage.cuda(deviceId)))
            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)

    for idx, param in enumerate(model.parameters()):
        last_model_tensors.append(copy.deepcopy(param.data))

    print('Begin!')
    logging.info('\n' + repr(args) + '\n')

    learning_rate = args.learning_rate

    testResults = [0, 0, 0, 1]
    # first run a forward pass
    # test_loss, acc, acc_5, testResults = test_model(rank, model, test_data, criterion=criterion, tokenizer=tokenizer)
    uploadEpoch = 0
    models_dir = None
    sorted_models_dir = None
    isComplete = False

    collate_fn = None

    if args.task == 'nlp':
        collate_fn = collate
    elif args.task == 'voice':
        collate_fn = voice_collate_fn

    last_test = time.time()

    if args.read_models_path:
        models_dir = scan_models(args.model_path)
        sorted_models_dir = sorted(models_dir)

    tempModelPath = logDir+'/model_'+str(args.this_rank)+'.pth.tar'

    for epoch in range(1, int(args.epochs) + 1):
        try:
            if epoch % args.decay_epoch == 0:
                learning_rate = max(args.min_learning_rate, learning_rate * args.decay_factor)

            trainedModels = []
            preTrainedLoss = []
            trainedSize = []
            trainSpeed = []
            virtualClock = []
            ranClients = []

            computeStart = time.time()

            if not args.test_only:
                # dump a copy of model
                with open(tempModelPath, 'wb') as fout:
                    pickle.dump(model, fout)

                for idx, nextClientId in enumerate(nextClientIds):
                    # roll back to the global model for simulation
                    with open(tempModelPath, 'rb') as fin:
                        model = pickle.load(fin)

                    _model_param, _loss, _trained_size, _speed, _time, _isSuccess = run_client(
                                clientId=nextClientId,
                                cmodel=model,
                                learning_rate=learning_rate,
                                iters=args.upload_epoch,
                                argdicts={'iters': epoch}
                            )

                    if _isSuccess is False:
                        continue

                    score = -1
                    if args.forward_pass:
                        forward_dataset = select_dataset(nextClientId, global_trainDB, batch_size=args.test_bsz, isTest=True)
                        forward_loss = run_forward_pass(model, forward_dataset)
                        score = forward_loss

                    trainedModels.append(_model_param)
                    preTrainedLoss.append(_loss if score == -1 else score)
                    trainedSize.append(_trained_size)
                    trainSpeed.append(_speed)
                    virtualClock.append(_time)
                    ranClients.append(nextClientId)

                #gc.collect()
            else:
                logging.info('====Start test round {}'.format(epoch))

                model_load_path = None
                # force to read models
                if models_dir is not None:
                    model_load_path = models_dir[sorted_models_dir[0]]

                    with open(model_load_path, 'rb') as fin:
                        model = pickle.load(fin)
                        model = model.to(device=device)

                    logging.info(f"====Now load model checkpoint: {sorted_models_dir[0]}")

                    del sorted_models_dir[0]

                    if len(sorted_models_dir) == 0:
                        isComplete = True

                testResults = [0., 0., 0., 1.]
                # designed for testing only
                for nextClientId in nextClientIds:

                    client_dataset = select_dataset(nextClientId, global_trainDB, batch_size=args.test_bsz, isTest=True,
                                                        fractional=False, collate_fn=collate_fn
                                                    )
                    test_loss, acc, acc_5, temp_testResults = test_model(rank, model, client_dataset, criterion=criterion, tokenizer=tokenizer)

                    logging.info(f'====Epoch: {epoch}, clientId: {nextClientId}, len: {temp_testResults[-1]}, test_loss: {test_loss}, acc: {acc}, acc_5: {acc_5}')

                    # merge test results
                    for idx, item in enumerate(temp_testResults):
                        testResults[idx] += item

                uploadEpoch = epoch

            computeEnd = time.time() - computeStart

            # upload the weight
            sendStart = time.time()
            testResults.append(uploadEpoch)
            queue.put({rank: [trainedModels, preTrainedLoss, trainedSize, isComplete, ranClients, trainSpeed, testResults, virtualClock]})
            uploadEpoch = -1
            sendDur = time.time() - sendStart

            logging.info("====Pushing takes {} s".format(sendDur))
            # wait for new models
            receStart = time.time()

            last_model_tensors = []
            for idx, param in enumerate(model.parameters()):
                tmp_tensor = torch.zeros_like(param.data)
                dist.broadcast(tensor=tmp_tensor, src=0)
                param.data = tmp_tensor
                last_model_tensors.append(copy.deepcopy(tmp_tensor))

            # receive current minimum step, and the clientIdLen for next training
            step_tensor = torch.zeros([world_size], dtype=torch.int).to(device=device)
            dist.broadcast(tensor=step_tensor, src=0)
            globalMinStep = step_tensor[0].item()
            totalLen = step_tensor[-1].item()
            endIdx = step_tensor[args.this_rank].item()
            startIdx = 0 if args.this_rank == 1 else step_tensor[args.this_rank - 1].item()

            clients_tensor = torch.zeros([totalLen], dtype=torch.int).to(device=device)
            dist.broadcast(tensor=clients_tensor, src=0)
            nextClientIds = [clients_tensor[x].item() for x in range(startIdx, endIdx)]

            receDur = time.time() - receStart

            #logging.info("====Finish receiving ps")
            evalStart = time.time()
            # test the model if necessary
            if epoch % int(args.eval_interval) == 0:
                model = model.to(device=device)
                # forward pass of the training data
                if args.test_train_data:
                    rank_train_data = select_dataset(
                                        args.this_rank, global_trainDB, batch_size=args.test_bsz, is_rank=rank,
                                        collate_fn=collate_fn
                                      )
                    test_loss, acc, acc_5, testResults = test_model(rank, model, rank_train_data, criterion=criterion, tokenizer=tokenizer)
                else:
                    test_loss, acc, acc_5, testResults = test_model(rank, model, global_testDB, criterion=criterion, tokenizer=tokenizer)

                logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {}, test_5_accuracy {} \n"
                            .format(epoch, round(time.time() - startTime, 4), round(time.time() - evalStart, 4), test_loss, acc, acc_5))

                uploadEpoch = epoch
                last_test = time.time()
                gc.collect()

            if epoch % args.dump_epoch == 0 and args.this_rank == 1:
                model = model.to(device='cpu')
                with open(logDir+'/'+str(args.model)+'_'+str(epoch)+'.pth.tar', 'wb') as fout:
                    pickle.dump(model, fout)

                logging.info("====Dump model successfully")
                model = model.to(device=device)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("====Error: {}, {}, {}, {}".format(e, exc_type, fname, exc_tb.tb_lineno))
            break

        if stop_flag.value:
            break

    queue.put({rank: [None, None, None, True, -1, -1]})
    logging.info("Worker {} has completed epoch {}!".format(args.this_rank, epoch))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def initiate_channel():
    BaseManager.register('get_queue')
    BaseManager.register('get_param')
    BaseManager.register('get_stop_signal')
    manager = BaseManager(address=(args.ps_ip, args.manager_port), authkey=b'queue')

    return manager

if __name__ == "__main__":
    #global global_trainDB, global_testDB

    setup_seed(args.this_rank)

    manager = initiate_channel()
    manager.connect()

    q = manager.get_queue()  # queue for parameter_server signal process
    param_q = manager.get_param()  # init
    stop_signal = manager.get_stop_signal()  # stop

    logging.info("====Start to initialize dataset")

    gc.disable()
    model, train_dataset, test_dataset = init_dataset()
    gc.enable()
    gc.collect()

    splitTrainRatio = []
    client_cfg = {}

    # Initialize PS - client communication channel
    world_size = len(workers) + 1
    this_rank = args.this_rank
    dist.init_process_group(args.backend, rank=this_rank, world_size=world_size)

    # Split the dataset
    # total_worker != 0 indicates we create more virtual clients for simulation
    if args.total_worker > 0 and args.duplicate_data == 1:
        workers = [i for i in range(1, args.total_worker + 1)]

    # load data partitioner (entire_train_data)
    dataConf = os.path.join(args.data_dir, 'sampleConf') if args.data_set == 'imagenet' else None

    logging.info("==== Starting training data partitioner =====")
    global_trainDB = DataPartitioner(data=train_dataset, splitConfFile=dataConf,
                        numOfClass=args.num_class, dataMapFile=args.data_mapfile)
    logging.info("==== Finished training data partitioner =====")

    dataDistribution = [int(x) for x in args.sequential.split('-')]
    distributionParam = [float(x) for x in args.zipf_alpha.split('-')]

    for i in range(args.duplicate_data):
        partition_dataset(global_trainDB, workers, splitTrainRatio, dataDistribution[i],
                                    filter_class=args.filter_class, arg = {'balanced_client':0, 'param': distributionParam[i]})
    global_trainDB.log_selection()

    report_data_info(this_rank, q)
    splitTestRatio = []

    logging.info("==== Starting testing data partitioner =====")
    testsetPartitioner = DataPartitioner(data=test_dataset, isTest=True, numOfClass=args.num_class)
    logging.info("==== Finished testing data partitioner =====")

    collate_fn = None

    if args.task == 'nlp':
        collate_fn = collate
    elif args.task == 'voice':
        collate_fn = voice_collate_fn

    partition_dataset(testsetPartitioner, [i for i in range(world_size-1)], splitTestRatio)
    global_testDB = select_dataset(this_rank, testsetPartitioner, batch_size=args.test_bsz, isTest=True, collate_fn=collate_fn)

    stop_flag = Value(c_bool, False)

    # no need to keep the raw data
    del train_dataset, test_dataset

    init_myprocesses(this_rank, world_size, model,
                                          q, param_q, stop_flag,
                                          run, args.backend, client_cfg)
