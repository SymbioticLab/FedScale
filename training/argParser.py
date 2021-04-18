import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_name', type=str, default='kuiper_job')
parser.add_argument('--log_path', type=str, default='../', help="default path is ../log")

# The basic configuration of the cluster
parser.add_argument('--ps_ip', type=str, default='127.0.0.1')
parser.add_argument('--ps_port', type=str, default='29501')
parser.add_argument('--manager_port', type=int, default='9005')
parser.add_argument('--this_rank', type=int, default=1)
parser.add_argument('--learners', type=str, default='1-2-3-4')
parser.add_argument('--total_worker', type=int, default=0)
parser.add_argument('--duplicate_data', type=int, default=1)
parser.add_argument('--data_mapfile', type=str, default=None)
parser.add_argument('--to_device', type=str, default='cuda')
parser.add_argument('--time_stamp', type=str, default='logs')
parser.add_argument('--task', type=str, default='cv')
parser.add_argument('--pacer_delta', type=float, default=5)
parser.add_argument('--pacer_step', type=int, default=20)
parser.add_argument('--capacity_bin', type=bool, default=True)
parser.add_argument('--exploration_alpha', type=float, default=0.3)
parser.add_argument('--exploration_factor', type=float, default=0.9)
parser.add_argument('--exploration_decay', type=float, default=0.95)
parser.add_argument('--fixed_clients', type=bool, default=False)
parser.add_argument('--user_trace', type=str, default=None)
parser.add_argument('--release_cache', type=bool, default=False)
parser.add_argument('--clock_factor', type=float, default=2.5, help="Refactor the clock time given the profile")

# The configuration of model and dataset
parser.add_argument('--data_dir', type=str, default='~/cifar10/')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--client_path', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0')
parser.add_argument('--read_models_path', type=bool, default=False)
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--score_mode', type=str, default='loss')
parser.add_argument('--proxy_avg', type=bool, default=False)
parser.add_argument('--proxy_mu', type=float, default=0.1)
parser.add_argument('--filter_less', type=int, default=32)
parser.add_argument('--filter_more', type=int, default=1e5)
parser.add_argument('--forward_pass', type=bool, default=False)
parser.add_argument('--run_all', type=bool, default=False)
parser.add_argument('--sampler_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--conf_path', type=str, default='~/dataset/')
parser.add_argument('--max_iter_store', type=int, default=100)
parser.add_argument('--overcommit', type=float, default=1.1)
parser.add_argument('--model_size', type=float, default=65536)
parser.add_argument('--sample_window', type=float, default=5.0)
parser.add_argument('--round_threshold', type=float, default=10)
parser.add_argument('--round_penalty', type=float, default=2.0)
parser.add_argument('--test_only', type=bool, default=False)
parser.add_argument('--malicious_clients', type=float, default=0)
parser.add_argument('--clip_bound', type=float, default=0.98)
parser.add_argument('--blacklist_rounds', type=int, default=-1)
parser.add_argument('--blacklist_max_len', type=float, default=0.3)
parser.add_argument('--noise_factor', type=float, default=0)


# The configuration of different hyper-parameters for training
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--test_bsz', type=int, default=256)
parser.add_argument('--heterogeneity', type=float, default=1.0)
parser.add_argument('--hetero_allocation', type=str, default='1.0-1.0-1.0-1.0-1.0-1.0')
parser.add_argument('--backend', type=str, default="nccl")
parser.add_argument('--display_step', type=int, default=20)
parser.add_argument('--upload_epoch', type=int, default=20)
parser.add_argument('--validate_interval', type=int, default=999999)
parser.add_argument('--stale_threshold', type=int, default=0)
parser.add_argument('--sleep_up', type=int, default=0)
parser.add_argument('--force_read', type=bool, default=False)
parser.add_argument('--test_interval', type=int, default=20)
parser.add_argument('--resampling_interval', type=int, default=1)
parser.add_argument('--sequential', type=str, default='0')
parser.add_argument('--single_sim', type=int, default=0)
parser.add_argument('--filter_class', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.04)
parser.add_argument('--model_avg', type=bool, default=True)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--dump_epoch', type=int, default=1000)
parser.add_argument('--decay_factor', type=float, default=0.95)
parser.add_argument('--decay_epoch', type=float, default=5)
parser.add_argument('--threads', type=str, default=4)
parser.add_argument('--num_loaders', type=int, default=2)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--eval_interval_prior', type=int, default=9999999)
parser.add_argument('--gpu_device', type=int, default=0)
parser.add_argument('--zipf_alpha', type=str, default='5')
parser.add_argument('--timeout', type=float, default=9999999)
parser.add_argument('--full_gradient_interval', type=int, default=20)
parser.add_argument('--is_even_avg', type=bool, default=True)
parser.add_argument('--sample_seed', type=int, default=233) #123 #233
parser.add_argument('--test_train_data', type=bool, default=False)
parser.add_argument('--enforce_random', type=bool, default=False)
parser.add_argument('--test_ratio', type=float, default=1.0)
parser.add_argument('--min_learning_rate', type=float, default=1e-4)
parser.add_argument('--loss_decay', type=float, default=0.2)
parser.add_argument('--skip_partition', type=bool, default=False)
parser.add_argument('--exploration_min', type=float, default=0.2)
parser.add_argument('--cut_off_util', type=float, default=0.7)

# for text clf
parser.add_argument('--clf_block_size', type=int, default=100)

# for yogi
parser.add_argument('--gradient_policy', type=str, default='')
parser.add_argument('--yogi_eta', type=float, default=5e-3)
parser.add_argument('--yogi_tau', type=float, default=1e-3)
parser.add_argument('--yogi_beta', type=float, default=0.999)
parser.add_argument('--yogi_beta2', type=float, default=-1)

# for albert

parser.add_argument(
    "--train_data_file", default='', type=str, help="The input training data file (a text file)."
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--model_type", type=str, default='', help="The model architecture to be trained or fine-tuned.",
)

# Other parameters
parser.add_argument(
    "--eval_data_file",
    default='',
    type=str,
    help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
)
parser.add_argument(
    "--line_by_line",
    action="store_true",
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument(
    "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
)

parser.add_argument(
    "--mlm", type=bool, default=True, help="Train with masked-language modeling loss instead of language modeling."
)
parser.add_argument(
    "--mlm_probability", type=float, default=0.1, help="Ratio of tokens to mask for masked language modeling loss"
)

parser.add_argument(
    "--config_name",
    default=None,
    type=str,
    help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
)
parser.add_argument(
    "--tokenizer_name",
    default=None,
    type=str,
    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
)
parser.add_argument(
    "--cache_dir",
    default=None,
    type=str,
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
)
parser.add_argument(
    "--block_size",
    default=64,
    type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
)

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
# parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=None,
    help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
)
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
)
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)
parser.add_argument(
    "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

# for tag prediction
parser.add_argument("--vocab_token_size", type=int, default=10000, help="For vocab token size")
parser.add_argument("--vocab_tag_size", type=int, default=500, help="For vocab tag size")

# for speech
parser.add_argument("--num_classes", type=int, default=35, help="For number of classes in speech")


# for voice
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/test_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=256, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=7, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                    help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')

args = parser.parse_args()



datasetCategories = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                    }

# Profiled relative speech w.r.t. Mobilenet
model_factor = {'shufflenet': 0.0644/0.0554,
    'albert': 0.335/0.0554,
    'resnet': 0.135/0.0554,
}

args.num_class = datasetCategories[args.data_set] if args.data_set in datasetCategories else 10
for model_name in model_factor:
    if model_name in args.model:
        args.clock_factor = args.clock_factor * model_factor[model_name]
        break
