import argparse

from fedscale.cloud import commons

parser = argparse.ArgumentParser()
parser.add_argument("--job_name", type=str, default="demo_job")
parser.add_argument("--log_path", type=str, default="./", help="default path is ../log")
parser.add_argument(
    "--wandb_token", type=str, default="", help="API key for wandb as login credentials"
)

# The basic configuration of the cluster
parser.add_argument("--ps_ip", type=str, default="127.0.0.1")
parser.add_argument("--ps_port", type=str, default="29500")
parser.add_argument("--this_rank", type=int, default=1)
parser.add_argument("--connection_timeout", type=int, default=60)
parser.add_argument("--experiment_mode", type=str, default=commons.SIMULATION_MODE)
parser.add_argument(
    "--engine",
    type=str,
    default=commons.PYTORCH,
    help="Tensorflow or Pytorch for cloud aggregation",
)
parser.add_argument("--num_executors", type=int, default=1)
parser.add_argument(
    "--executor_configs", type=str, default="127.0.0.1:[1]"
)  # seperated by ;
# Note: In async mode, the num_participants param is treated as the async buffer size. In sync, this is the number
# of clients that are selected each round.
parser.add_argument("--num_participants", type=int, default=4)
parser.add_argument("--data_map_file", type=str, default=None)
parser.add_argument("--use_cuda", type=str, default="True")
parser.add_argument("--cuda_device", type=str, default=None)
parser.add_argument("--time_stamp", type=str, default="logs")
parser.add_argument("--task", type=str, default="cv")
parser.add_argument("--device_avail_file", type=str, default=None)
parser.add_argument(
    "--clock_factor",
    type=float,
    default=1.0,
    help="Refactor the clock time given the profile",
)

# The configuration of model and dataset
parser.add_argument(
    "--model_zoo",
    type=str,
    default="torchcv",
    help="model zoo to load the models from",
    choices=["torchcv", "fedscale-torch-zoo", "fedscale-tensorflow-zoo"],
)
parser.add_argument("--data_dir", type=str, default="~/cifar10/")
parser.add_argument("--device_conf_file", type=str, default="/tmp/client.cfg")
parser.add_argument("--model", type=str, default="shufflenet_v2_x2_0")
parser.add_argument("--data_set", type=str, default="cifar10")
parser.add_argument("--sample_mode", type=str, default="random")
parser.add_argument("--filter_less", type=int, default=32)
parser.add_argument("--filter_more", type=int, default=1e15)
parser.add_argument("--train_uniform", type=bool, default=False)
parser.add_argument("--conf_path", type=str, default="~/dataset/")
parser.add_argument("--overcommitment", type=float, default=1.3)
parser.add_argument("--model_size", type=float, default=65536)
parser.add_argument("--round_threshold", type=float, default=30)
parser.add_argument("--round_penalty", type=float, default=2.0)
parser.add_argument("--clip_bound", type=float, default=0.9)
parser.add_argument("--blacklist_rounds", type=int, default=-1)
parser.add_argument("--blacklist_max_len", type=float, default=0.3)
parser.add_argument("--embedding_file", type=str, default="glove.840B.300d.txt")
parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 28, 28])
parser.add_argument("--save_checkpoint", type=bool, default=False)


# The configuration of different hyper-parameters for training
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--local_steps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--test_bsz", type=int, default=128)
parser.add_argument("--backend", type=str, default="gloo")
parser.add_argument("--learning_rate", type=float, default=5e-2)
parser.add_argument("--min_learning_rate", type=float, default=5e-5)
parser.add_argument("--input_dim", type=int, default=0)
parser.add_argument("--output_dim", type=int, default=0)
parser.add_argument("--dump_epoch", type=int, default=1e10)
parser.add_argument("--decay_factor", type=float, default=0.98)
parser.add_argument("--decay_round", type=float, default=10)
parser.add_argument("--num_loaders", type=int, default=2)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--sample_seed", type=int, default=233)  # 123 #233
parser.add_argument("--test_ratio", type=float, default=1.0)
parser.add_argument("--loss_decay", type=float, default=0.2)
parser.add_argument("--exploration_min", type=float, default=0.3)
parser.add_argument("--cut_off_util", type=float, default=0.05)  # 95 percentile

parser.add_argument("--gradient_policy", type=str, default=None)

# for yogi
parser.add_argument("--yogi_eta", type=float, default=3e-3)
parser.add_argument("--yogi_tau", type=float, default=1e-8)
parser.add_argument("--yogi_beta", type=float, default=0.9)
parser.add_argument("--yogi_beta2", type=float, default=0.99)

# for q-fedavg
parser.add_argument("--qfed_q", type=float, default=1.0)


# for prox
parser.add_argument("--proxy_mu", type=float, default=0.1)

# for detection
parser.add_argument("--cfg_file", type=str, default="./utils/rcnn/cfgs/res101.yml")
parser.add_argument("--test_output_dir", type=str, default="./logs/server")
parser.add_argument("--train_size_file", type=str, default="")
parser.add_argument("--test_size_file", type=str, default="")
parser.add_argument("--data_cache", type=str, default="")
parser.add_argument("--backbone", type=str, default="./resnet50.pth")


# for malicious
parser.add_argument("--malicious_factor", type=int, default=1e15)

# for asynchronous FL
parser.add_argument("--max_concurrency", type=int, default=10)
parser.add_argument("--max_staleness", type=int, default=5)

# for differential privacy
parser.add_argument("--noise_factor", type=float, default=0.1)
parser.add_argument("--clip_threshold", type=float, default=3.0)
parser.add_argument("--target_delta", type=float, default=0.0001)

# for Oort
parser.add_argument("--pacer_delta", type=float, default=5)
parser.add_argument("--pacer_step", type=int, default=20)
parser.add_argument("--exploration_alpha", type=float, default=0.3)
parser.add_argument("--exploration_factor", type=float, default=0.9)
parser.add_argument("--exploration_decay", type=float, default=0.98)
parser.add_argument("--sample_window", type=float, default=5.0)

# for albert
parser.add_argument(
    "--line_by_line",
    action="store_true",
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument("--clf_block_size", type=int, default=32)


parser.add_argument(
    "--mlm",
    type=bool,
    default=False,
    help="Train with masked-language modeling loss instead of language modeling.",
)
parser.add_argument(
    "--mlm_probability",
    type=float,
    default=0.15,
    help="Ratio of tokens to mask for masked language modeling loss",
)
parser.add_argument(
    "--overwrite_cache",
    type=bool,
    default=False,
    help="Overwrite the cached training and evaluation sets",
)
parser.add_argument(
    "--block_size",
    default=64,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)


parser.add_argument(
    "--weight_decay", default=0, type=float, help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)

# for tag prediction
parser.add_argument(
    "--vocab_token_size", type=int, default=10000, help="For vocab token size"
)
parser.add_argument(
    "--vocab_tag_size", type=int, default=500, help="For vocab tag size"
)

# for rl example
parser.add_argument("--epsilon", type=float, default=0.9, help="greedy policy")
parser.add_argument("--gamma", type=float, default=0.9, help="reward discount")
parser.add_argument("--memory_capacity", type=int, default=2000, help="memory capacity")
parser.add_argument(
    "--target_replace_iter", type=int, default=15, help="update frequency"
)
parser.add_argument("--n_actions", type=int, default=2, help="action number")
parser.add_argument("--n_states", type=int, default=4, help="state number")


parser.add_argument(
    "--num_classes", type=int, default=35, help="For number of classes of the dataset"
)


# for voice
parser.add_argument(
    "--train-manifest",
    metavar="DIR",
    help="path to train manifest csv",
    default="data/train_manifest.csv",
)
parser.add_argument(
    "--test-manifest",
    metavar="DIR",
    help="path to test manifest csv",
    default="data/test_manifest.csv",
)
parser.add_argument("--sample-rate", default=16000, type=int, help="Sample rate")
parser.add_argument(
    "--labels-path",
    default="labels.json",
    help="Contains all characters for transcription",
)
parser.add_argument(
    "--window-size",
    default=0.02,
    type=float,
    help="Window size for spectrogram in seconds",
)
parser.add_argument(
    "--window-stride",
    default=0.01,
    type=float,
    help="Window stride for spectrogram in seconds",
)
parser.add_argument(
    "--window", default="hamming", help="Window type for spectrogram generation"
)
parser.add_argument("--hidden-size", default=256, type=int, help="Hidden size of RNNs")
parser.add_argument("--hidden-layers", default=7, type=int, help="Number of RNN layers")
parser.add_argument(
    "--rnn-type", default="lstm", help="Type of the RNN. rnn|gru|lstm are supported"
)
parser.add_argument(
    "--finetune",
    dest="finetune",
    action="store_true",
    help='Finetune the model from checkpoint "continue_from"',
)
parser.add_argument(
    "--speed-volume-perturb",
    dest="speed_volume_perturb",
    action="store_true",
    help="Use random tempo and gain perturbations.",
)
parser.add_argument(
    "--spec-augment",
    dest="spec_augment",
    action="store_true",
    help="Use simple spectral augmentation on mel spectograms.",
)
parser.add_argument(
    "--noise-dir",
    default=None,
    help="Directory to inject noise into audio. If default, noise Inject not added",
)
parser.add_argument(
    "--noise-prob", default=0.4, help="Probability of noise being added per sample"
)
parser.add_argument(
    "--noise-min",
    default=0.0,
    help="Minimum noise level to sample from. (1.0 means all noise, not original signal)",
    type=float,
)
parser.add_argument(
    "--noise-max",
    default=0.5,
    help="Maximum noise levels to sample from. Maximum 1.0",
    type=float,
)
parser.add_argument(
    "--no-bidirectional",
    dest="bidirectional",
    action="store_false",
    default=True,
    help="Turn off bi-directional RNNs, introduces lookahead convolution",
)

args, unknown = parser.parse_known_args()
args.use_cuda = eval(args.use_cuda)


datasetCategories = {
    "Mnist": 10,
    "cifar10": 10,
    "imagenet": 1000,
    "emnist": 47,
    "openImg": 596,
    "google_speech": 35,
    "femnist": 62,
    "yelp": 5,
}

# Profiled relative speech w.r.t. Mobilenet
model_factor = {
    "shufflenet": 0.0644 / 0.0554,
    "albert": 0.335 / 0.0554,
    "resnet": 0.135 / 0.0554,
}

args.num_class = datasetCategories.get(args.data_set, args.num_classes)
for model_name in model_factor:
    if model_name in args.model:
        args.clock_factor = args.clock_factor * model_factor[model_name]
        break
