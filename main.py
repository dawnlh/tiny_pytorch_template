import os
import argparse
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.multiprocessing as mp
from utils.utils import Logger
from models.builder import build_model
from train import _train
from test import _test



# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(cfg):
    # make dirs
    os.makedirs(cfg['output_dir'], exist_ok=True)
    if cfg.mode == 'train':
        os.makedirs(cfg['model_save_dir'], exist_ok=True)
        os.makedirs(cfg['event_dir'], exist_ok=True)

    # get logger
    logger = Logger(log_name='main', log_file=cfg['log_path'])
    logger.info(OmegaConf.to_yaml(cfg))

    # build model
    model = build_model(cfg,logger)
    logger.info(model)

    # run
    if cfg.mode == 'train':
        if cfg.num_gpus > 1:
            mp.spawn(_train, nprocs=cfg.num_gpus,args=(model, cfg))
        else:
            _train(0, model, cfg)
    elif cfg.mode == 'test':
        _test(model, cfg)
    else:
        raise ValueError('mode must be train or test')

    # if torch.cuda.is_available():
    #     model.cuda()
    # if cfg.mode == 'train':
    #     _train(model, cfg, logger)

    # elif cfg.mode == 'test':
    #     _test(model, cfg, logger)


if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = parser.parse_args()

    # load config
    config_file = args.config
    cfg = OmegaConf.load(config_file)

    # set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg['gpu_ids'])
    cfg.num_gpus = len(cfg['gpu_ids'])

    # set dirs
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join('exp/', cfg.model.name, cfg.mode, date_time)
    cfg.work_dir = work_dir
    cfg.model_save_dir = os.path.join(work_dir, 'ckp/')
    cfg.output_dir = os.path.join(work_dir, 'output/')
    cfg.event_dir = os.path.join(work_dir, 'event/')
    cfg.log_path = os.path.join(work_dir, 'runtime.log')

    # save config
    os.makedirs(cfg.work_dir, exist_ok=True)
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    with open(os.path.join(cfg.work_dir, 'config.yaml'), 'w') as f:
        f.write(config_yaml)

    main(cfg)
