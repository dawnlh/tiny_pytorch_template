import os
import torch

from data.data_load import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


def _train(model, args, logger=None):
    # ---------------------------- init ------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.event_dir)
    epoch = 1
    best_psnr=-1
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    

    # ---------------------------- loss ------------------------------------
    criterion = torch.nn.L1Loss()
    epoch_loss_adder = Adder()
    iter_loss_adder = Adder()
    
    
    # ----------------------- optimizer & scheduler ---------------------------------
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    

    # ---------------------------- dataloader ------------------------------------
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    
    
    # ---------------------------- resume ------------------------------------
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        logger.info('💡 Resume from epoch %d'%epoch)
        epoch += 1

    
    # ---------------------------- train ------------------------------------
    for epoch_idx in range(epoch, args.num_epoch + 1):
        # reset timer
        epoch_timer.tic()
        iter_timer.tic()

        # run iter
        for iter_idx, batch_data in enumerate(dataloader):
            # data to device
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            # model forward
            pred_img = model(input_img)
            
            # loss
            loss = criterion(pred_img, label_img)
            iter_loss_adder(loss.item())
            epoch_loss_adder(loss.item())

            # backward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -- iter ending --
            if (iter_idx + 1) % args.logger_freq == 0:
                # logger
                lr = check_lr(optimizer)
                logger.info("[ Iter %04d/%04d ]\tTime: %7.4f min, LR: %.10f, Loss: %7.4f" % (
                    iter_idx + 1, max_iter, iter_timer.toc(), lr, iter_loss_adder.average()))
                writer.add_scalar('Train/Loss', iter_loss_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                # reset
                iter_timer.tic()
                iter_loss_adder.reset()
        
        # -- epoch ending --   
        # save model    
        overwrite_name = os.path.join(args.model_save_dir, 'model_latest.pth')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%03d.pth' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        logger.info("[ Epoch %04d/%04d ]\tTime: %4.2f min, Loss: %7.4f" % (
            epoch_idx, args.num_epoch, epoch_timer.toc(), epoch_loss_adder.average()))
        
        # update & reset
        epoch_loss_adder.reset()
        scheduler.step()

        # valid
        if epoch_idx % args.valid_freq == 0:
            val_results = _valid(model, args, epoch_idx)
            logger.info('\t==> Average PSNR %.2f dB\n' % (val_results))
            writer.add_scalar('Valid/PSNR', val_results, epoch_idx)
            # update best model
            if val_results >= best_psnr:
                torch.save({'model': model.state_dict(),'epoch': epoch_idx}, os.path.join(args.model_save_dir, 'model_best.pth'))
                best_psnr = val_results
    # -- train ending --
    save_name = os.path.join(args.model_save_dir, 'Final.pth')
    torch.save({'model': model.state_dict(),'epoch': epoch_idx}, save_name)
