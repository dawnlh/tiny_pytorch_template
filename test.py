import os
import torch
from torchvision.utils import save_image
from utils.utils import Logger, Adder
from data.builder import build_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time


def _test(model, args, logger=None):
    # ---------------------------- init ------------------------------------
    if args.test_on_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    logger = Logger(log_name='test', log_file=args['log_path'])
    logger.info('==> Test on device: {}'.format(device))

    # ---------------------------- dataloader ------------------------------------
    dataloader = build_dataloader(args, mode='test')

    # ---------------------------- model ------------------------------------
    model.to(device)

    # ---------------------------- eval ------------------------------------
    model.eval()
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

            if args.save_test_image:
                save_name = os.path.join(args.output_dir, name[0])
                pred_clip += 0.5 / 255
                save_image(pred, save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            logger.info('[ Iter %04d/%04d ] PSNR: %.2f dB, time: %.2f s' % (iter_idx + 1, len(dataloader), psnr, cur_time - start_time))

        logger.info('==========================================================')
        logger.info('Aver. PSNR %.2f dB, Total time: %.2f min' % (psnr_adder.average(), (cur_time - eval_start_time)/60))
