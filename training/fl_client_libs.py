# package for client
from flLibs import *

# logDir = os.path.join(args.log_path, 'logs', args.job_name, args.time_stamp, 'worker')
# logFile = os.path.join(logDir, 'log_'+str(args.this_rank))
import os

logDir = os.path.join(os.environ['HOME'], "models", args.model, args.time_stamp, 'learner')
logFile = os.path.join(logDir, 'log_'+str(args.this_rank))

def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    logging.basicConfig(
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d:%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logFile, mode='a'),
                        logging.StreamHandler()
                    ])

def get_ps_ip():
    global args

    ip_file = os.path.join(logDir, '../aggregator/ip')
    ps_ip = None
    while not os.path.exists(ip_file):
        time.sleep(1)

    with open(ip_file, 'rb') as fin:
        ps_ip = pickle.load(fin)

    args.ps_ip = ps_ip
    logging.info('Config ps_ip on {}, args.ps_ip is {}'.format(ps_ip, args.ps_ip))


def initiate_client_setting():
    init_logging()
    get_ps_ip()

