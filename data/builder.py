from .dataloader import dataloader
from .dataset import DeblurDataset

def build_dataloader(cfg, mode='train', logger=None):
    # config
    dataloader_cfg = cfg[mode+'_dataloader']
    dataset_cfg = dataloader_cfg.pop('dataset')
    
    # dataset
    dataset_name = dataset_cfg.pop('name')
    dataset_cfg.update({'mode': mode})
    dataset_ = eval(dataset_name)(**dataset_cfg)
    
    # dataloader
    is_distributed = cfg.num_gpus > 1 and mode=='train'
    dataloader_ = dataloader(dataset=dataset_, is_distributed=is_distributed, **dataloader_cfg)
    return dataloader_
    