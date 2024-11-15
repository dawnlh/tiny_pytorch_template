import os
import torch
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from utils.utils import Adder


def _valid(model, val_dataloader, args, epoch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    psnr_adder = Adder()
    pbar = tqdm(total=len(val_dataloader), desc='Valid')

    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            pred = model(input_img)

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            pbar.update(1)
            if args.save_test_image:
                save_name = os.path.join(args.output_dir, f'ep_{epoch:04d}_{name[0]}')
                pred_clip += 0.5 / 255
                save_image(pred, save_name)
    model.train()
    pbar.close()

    return psnr_adder.average()
