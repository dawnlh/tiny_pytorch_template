import torch
import inspect
# from .lr_scheduler import LinearLR, VibrateLR, MultiStepRestartLR, CosineAnnealingRestartLR, CosineAnnealingRestartCyclicLR
# import importlib
from . import lr_scheduler, optimizer
from .warmup import WarmupLR

# =============== optimizer/lr_scheduler list ================
OPTIMS, SCHEDS = {}, {}

for module_name in dir(torch.optim)+dir(optimizer):
    if module_name.startswith('__'):
        continue
    try:
        _optim = getattr(torch.optim, module_name)
    except AttributeError:
        _optim = getattr(optimizer, module_name)
    if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
        OPTIMS.update({module_name: _optim})

# local_lr_scheduler = importlib.import_module('optimization.lr_scheduler')
for module_name in dir(torch.optim.lr_scheduler)+dir(lr_scheduler):
    if module_name.startswith('__'):
        continue
    try:
        _sched = getattr(torch.optim.lr_scheduler, module_name)
    except AttributeError:
        _sched = getattr(lr_scheduler, module_name)
    
    if inspect.isclass(_sched) and issubclass(
            _sched, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.LRScheduler)):
        SCHEDS.update({module_name: _sched})


# =============== build optimizer ================
def build_optimizer(params, cfg, logger=None):
    """
    build optimizer
    """
    optimizer_config = cfg['optimizer']
    optim_name = optimizer_config.pop('name')
    if optim_name in OPTIMS:
        return OPTIMS.get(optim_name)(params, **optimizer_config)
    else:
        raise NotImplementedError(f'{optim_name} is not implemented in the OPTIMS list')


# ================ build lr_scheduler ================
def build_lr_scheduler(optimizer, cfg, logger=None):
    """
    build scheduler
    """
    lr_scheduler_config = cfg['lr_scheduler']
    sched_name = lr_scheduler_config.pop('name')
    warmup_cfg = lr_scheduler_config.pop('warmup', None)
    if warmup_cfg is not None:
        init_lr = warmup_cfg.pop('init_lr', 1e-5)
        num_warmup = warmup_cfg.pop('epoch', -1)
        warmup_strategy = warmup_cfg.pop('strategy', 'linear')

    if sched_name in SCHEDS:
        scheduler_ =  SCHEDS.get(sched_name)(optimizer, **lr_scheduler_config)
        scheduler_ = WarmupLR(scheduler_, init_lr, num_warmup, warmup_strategy)
        return scheduler_
    else:
        raise NotImplementedError(f'{sched_name} is not implemented in the SCHEDS list')
