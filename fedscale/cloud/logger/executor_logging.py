from fedscale.cloud.fllibs import *
import fedscale.cloud.config_parser as parser

logDir = None


def init_logging():
    global logDir

    logDir = os.path.join(parser.args.log_path, "logs", parser.args.job_name,
                          parser.args.time_stamp, 'executor')
    logFile = os.path.join(logDir, 'log')
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
