import os
import torch
from omegaconf import OmegaConf
from models.UNet import build_net
from train import _train
from test import _test
from datetime import datetime
import numpy as np
from utils import Logger

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(cfg):
    # make dirs
    dirs = setup_directories(cfg)
    os.makedirs(dirs['output_dir'], exist_ok=True)
    if cfg.mode == 'train':
        os.makedirs(dirs['model_save_dir'], exist_ok=True)
        os.makedirs(dirs['event_dir'], exist_ok=True)

    # get logger
    logger = Logger(log_path=dirs['log_path'])
    logger.info(OmegaConf.to_yaml(cfg))

    # build model
    model = build_net(cfg.model_name)
    logger.info(model)

    # run
    if torch.cuda.is_available():
        model.cuda()
    if cfg.mode == 'train':
        _train(model, cfg, logger)

    elif cfg.mode == 'test':
        _eval(model, cfg, logger)

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # load config
    config_file = args.config
    cfg = OmegaConf.load(config_file)

    # set dirs
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = os.path.join('exp/', cfg.model_name, cfg.mode, date_time)
    cfg.work_dir = work_dir
    cfg.model_save_dir = os.path.join(work_dir, 'ckp/')
    cfg.output_dir = os.path.join(work_dir, 'output/')
    cfg.event_dir = os.path.join(work_dir, 'event/')
    cfg.log_path = os.path.join(oworkdir, 'runtime.log')   

    # save config
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    with open(os.path.join(cfg.work_dir, 'config.yaml'),'w') as f:
        f.write(config_yaml)

    main(cfg)