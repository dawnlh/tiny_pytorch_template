import os
import torch
import argparse
from models.UNet import build_net
from train import _train
from eval import _eval
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

def main(args):
    # make dirs
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == 'train':
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.event_dir, exist_ok=True)

    # get logger
    logger = Logger(log_path=args.log_path)
    logger.info(args)

    # build model
    model = build_net(args.model_name)
    logger.info(model)

    # run
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args, logger)

    elif args.mode == 'test':
        _eval(model, args, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='UNet', choices=['UNet'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--logger_freq', type=int, default=1)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # Test
    parser.add_argument('--test_model', type=str, default='weights/MIMO-UNet.pth')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    # main
    args = parser.parse_args()
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join('results/', args.model_name, args.mode, date_time)

    args.model_save_dir = os.path.join(output_dir, 'weights/')
    args.output_dir = os.path.join(output_dir, 'output/')
    args.event_dir = os.path.join(output_dir, 'event/')
    args.log_path = os.path.join(output_dir, 'runtime.log')
    
    main(args)
