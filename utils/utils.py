import time
import os
import platform, functools
from datetime import datetime
import logging, logging.config
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn.parallel import DataParallel, DistributedDataParallel

# ---- DDP ----
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def master_only(func):
    # decorator for running with master process in DDP
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper


def is_master():
    # check whether it is the master node
    return not dist.is_initialized() or dist.get_rank() == 0

def ddp_init(rank=0, num_gpus=1):
    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=num_gpus,
        rank=rank)

def ddp_finalize():
    dist.barrier()
    dist.destroy_process_group()
    return


def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

 
# ---- file operations ----

def file_traverse(dir, ext=None):
    """
    traverse all the files and get their paths
    Args:
        dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    data_paths = []
    skip_num = 0
    file_num = 0

    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            if ext and img_path.split('.')[-1] not in ext:
                print('Skip a file: %s' % (img_path))
                skip_num += 1
            else:
                data_paths.append(img_path)
                file_num += 1
    return sorted(data_paths), file_num, skip_num

def get_file_path(data_dir, ext=None):
    """
    Get file paths for given directory or directories

    Args:
        data_dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    if isinstance(data_dir, str):
        # single dataset
        data_paths, file_num, skip_num = file_traverse(data_dir, ext)
    elif isinstance(data_dir, list):
        # multiple datasets
        data_paths, file_num, skip_num = [], 0, 0
        for data_dir_n in sorted(data_dir):
            data_paths_n, file_num_n, skip_num_n = file_traverse(
                data_dir_n, ext)
            data_paths.extend(data_paths_n)
            file_num += file_num_n
            skip_num += skip_num_n
    else:
        raise ValueError('data dir should be a str or a list of str')

    return sorted(data_paths), file_num, skip_num


# ---- logging  ----
class Logger():
    def __init__(self, log_name=None, log_file='./runtime.log', log_level=logging.INFO, rank=0):
        logger = logging.getLogger(log_name)
        # if the logger has been initialized, just return it
        if log_name != 'metric':
            format_str = '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s'
        else:
            format_str = ''
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(stream_handler)
        logger.propagate = False
        if rank != 0:
            logger.setLevel('ERROR')
        elif log_file is not None:
            logger.setLevel(log_level)
            file_handler = logging.FileHandler(log_file, 'a')
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        self.logger = logger

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
            self.logger.critical(msg, *args, **kwargs)
    def fatal(self, msg, *args, **kwargs):
            self.logger.fatal(msg, *args, **kwargs)

    def debug(self, msg,*args, **kwargs):
            self.logger.debug(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
            self.logger.exception(msg, *args, **kwargs)

class TensorboardWriter():
    def __init__(self, log_dir, enabled=True):
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_hparams'
        }
        self.timer = datetime.now()

    def set_step(self, step, speed_chk=None):  # phases = 'train'|'valid'|None
        self.step = step
        # measure the calculation speed by call this function between 2 steps (steps_per_sec)
        if speed_chk and step != 0:
            duration = datetime.now() - self.timer
            self.add_scalar(f'steps_per_sec/{speed_chk}',
                            1 / duration.total_seconds())
        self.timer = datetime.now()


    def writer_update(self, step, phase, metrics, image_tensors=None):
        # hook after iter
        self.set_step(step, speed_chk=f'{phase}')

        metric_str = ''
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.add_scalar(f'{phase}/{k}', v)
                metric_str += f'{k}: {v:8.5f} '

        if image_tensors:
            for k, v in image_tensors.items():
                self.add_image(
                    f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

        return metric_str  # metric string for logger

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(var1, var2, *args, **kwargs):
                if add_data is not None:
                    add_data(var1, var2, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name, None)
            if not attr:
                raise AttributeError('unimplemented attribute')
            return attr


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
        self.deviders = {'s': 1, 'm': 60, 'h': 3600}

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

    def eta(self, total_iter, cur_iter, option=None):
        option = option if option else self.option
        return (total_iter - cur_iter) * self.total(option) / cur_iter
   
# --- training ---

def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

@master_only
def save_checkpoint(model, dir, epoch, optimizer=None, scheduler=None, prefix=None):
    
    # prefix
    prefix = f'Epoch_{epoch:04d}' if prefix is None else prefix

    # param arrange
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_ = model.module
    else:
        model_ = model
    
    state_dict = model_.state_dict()
    for key, param in state_dict.items():
        if key.startswith('module.'):  # remove unnecessary 'module.'
            key = key[7:]
        state_dict[key] = param.cpu()

    # save weights
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict
    }, os.path.join(dir, prefix + '_model.pth'))

    # save statr
    if optimizer and scheduler:
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(dir, prefix + '_state.pth'))

