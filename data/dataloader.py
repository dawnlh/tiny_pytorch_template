from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def dataloader(dataset, sampler=None, batch_size=8, num_workers=8, shuffle=False, pin_memory=True, is_distributed=False):

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    if sampler:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    return dataloader