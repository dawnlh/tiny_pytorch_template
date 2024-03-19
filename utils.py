import time
import numpy as np
import logging, logging.config

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count
    
    def info(self):
        return (self.num, self.count)


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.init_tm = -1
        self.option = option
        self.deviders = {'s':1, 'm':60, 'h':3600}

    def tic(self):
        self.tm = time.time()
        if self.init_tm == -1:
            self.init_tm = self.tm

    def toc(self, option=None):
        option = option if option else self.option
        return (time.time() - self.tm) / self.deviders[option]

    def init(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.init_tm) )

    def total(self, option=None):
        option = option if option else self.option
        return (time.time() - self.init_tm) / self.deviders[option]

def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr



def Logger(name=None, log_path='./runtime.log'):
    config_dict = {
        "version": 1,
        "formatters": {
            "simple": {
            "format": "%(message)s"
            },
            "detailed": {
            "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            }
        },
        "handlers": {
            "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
            },
            "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": log_path
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "disable_existing_loggers": False
    }
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger 