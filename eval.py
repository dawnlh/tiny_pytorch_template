import os
import torch
from torchvision.utils import save_image
from utils import Adder
from data.data_load import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time


def _eval(model, args, logger=None):
    # ---------------------------- init ------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ---------------------------- dataloader ------------------------------------
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)

    # ---------------------------- load model ------------------------------------
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    model.eval()

    # ---------------------------- eval ------------------------------------
    with torch.no_grad():
        psnr_adder = Adder()
        
        # run iter
        eval_start_time = time.time()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            start_time = time.time()

            pred = model(input_img)

            cur_time = time.time()
            

            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.output_dir, name[0])
                pred_clip += 0.5 / 255
                save_image(pred, save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            logger.info('[ Iter %04d/%04d ] PSNR: %.2f dB, time: %.2f s' % (iter_idx + 1, len(dataloader), psnr, cur_time - start_time))

        logger.info('==========================================================')
        logger.info('Aver. PSNR %.2f dB, Total time: %.2f min' % (psnr_adder.average(), (cur_time - eval_start_time)/60))
