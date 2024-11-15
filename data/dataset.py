import os
from torch.utils.data import Dataset
from PIL import Image as Image
import numpy as np
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform={}, mode='train'):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'blur/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.is_test = mode == 'test'
        transform_list = []
        if transform:
            if 'random_crop' in transform:
                transform_list.append(transforms.RandomCrop(transform['random_crop']))
            if 'random_hflip' in transform:
                transform_list.append(transforms.RandomHorizontalFlip())
            if 'random_vflip' in transform:
                transform_list.append(transforms.RandomVerticalFlip())
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            torch_state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(torch_state)
            label = self.transform(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
