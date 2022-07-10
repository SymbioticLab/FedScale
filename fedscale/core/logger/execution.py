# package for client
import os

from fedscale.core.fllibs import *

logDir = os.path.join(args.log_path, "logs", args.job_name,
                      args.time_stamp, 'executor')
logFile = os.path.join(logDir, 'log')


def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logFile, mode='a'),
            logging.StreamHandler()
        ])


def initiate_client_setting():
    init_logging()


# initiate the log path, and executor ips
initiate_client_setting()
