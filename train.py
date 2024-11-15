import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from data.builder import build_dataloader
from loss.builder import build_loss
from optimization.builder import build_optimizer, build_lr_scheduler
from utils.utils import Logger, TensorboardWriter, Adder, Timer, save_checkpoint, ddp_init, ddp_finalize, collect
from valid import _valid


def _train(rank, model, args):

    # ---------------------------- init ------------------------------------
    # master = rank==0
    if args.num_gpus > 1:
        ddp_init(rank=rank, num_gpus=args.num_gpus)
    torch.cuda.set_device(rank)

    # ---------------------------- logger ------------------------------------
    epoch = 1
    best_psnr = -1
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    # logger = get_logger(log_name='train', log_file=args['log_path'])
    logger = Logger(log_name='train', log_file=args['log_path'], rank=rank)
    writer = TensorboardWriter(args.event_dir, rank == 0)

    # ---------------------------- model ------------------------------------
    device = torch.device('cuda:{:d}'.format(rank))
    model = model.to(device)
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[rank])

    # ---------------------------- loss ------------------------------------
    criterion = build_loss(args)
    epoch_loss_adder = Adder()
    iter_loss_adder = Adder()

    # ----------------------- optimizer & scheduler ---------------------------------
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n')

    optimizer = build_optimizer(trainable_params, args)
    scheduler = build_lr_scheduler(optimizer, args)
    # print('---', scheduler.get_last_lr()[0])
    

    # ---------------------------- dataloader ------------------------------------
    dataloader = build_dataloader(args, mode='train')
    max_iter = len(dataloader)
    val_dataloader = build_dataloader(args, mode='test')

    # ---------------------------- resume ------------------------------------
    if args.resume_state:
        state = torch.load(args.resume_state, weights_only=True)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        logger.info(f'==> Resume training state of Epoch-{epoch} from: {args.resume_state}')
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
                # lr = check_lr(optimizer)
                lr = scheduler.get_last_lr()[0]
                logger.info("[ Iter %04d/%04d ]\tTime: %4.2f min, LR: %.8e, Loss: %.6f" %
                            (iter_idx + 1, max_iter, iter_timer.toc(), lr, iter_loss_adder.average()))
                writer.writer_update(iter_idx + (epoch_idx - 1) * max_iter, 'train', {'loss': iter_loss_adder.average()})
                # reset
                iter_timer.tic()
                iter_loss_adder.reset()

        # -- epoch ending --
        # save model
        save_checkpoint(model, args.model_save_dir, epoch_idx, optimizer, scheduler, prefix='latest')
        if epoch_idx % args.save_freq == 0:
            save_checkpoint(model, args.model_save_dir, epoch_idx, optimizer, scheduler)
        logger.info("[ Epoch %04d/%04d ]\tTime: %4.2f min, Eta time: %4.2f h, Loss: %.6f" %
                    (epoch_idx, args.num_epoch, epoch_timer.toc(), epoch_timer.eta(args.num_epoch, epoch_idx,
                                                                                   'h'), epoch_loss_adder.average()))

        # update & reset
        epoch_loss_adder.reset()
        scheduler.step()

        # valid
        if epoch_idx % args.valid_freq == 0 and rank == 0:
            val_results = _valid(model, val_dataloader, args, epoch_idx)
            logger.info('Valid: Aver. PSNR %.2f dB\n' % (val_results))
            writer.writer_update(epoch_idx, 'valid', {'psnr': val_results})
            # update best model
            if val_results >= best_psnr:
                save_checkpoint(model, args.model_save_dir, epoch_idx, optimizer, scheduler, prefix='best')
                best_psnr = val_results

    # -- train ending --
    save_checkpoint(model, args.model_save_dir, epoch_idx, optimizer, scheduler, prefix='Final')
    if args.num_gpus > 1:
        ddp_finalize()
