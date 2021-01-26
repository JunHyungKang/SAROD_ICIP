import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import yolov5.test_original as test  # import test.py to get mAP after each epoch
import yolov5.test_rl as test_rl  # import test.py to get mAP after each epoch
from yolov5.models.yolo import Model
from yolov5.utils import google_utils
from yolov5.utils.datasets import *
from yolov5.utils.utils import *
from yolov5.models.experimental import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

class yolov5():

    def __init__(self, opt_tr, opt_eval):
        self.opt = opt_tr
        self.opt_eval = opt_eval

        self.wdir = 'weights' + os.sep  # weights dir
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = 'results.txt'

        # Hyperparameters
        self.hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
                    'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'momentum': 0.937,  # SGD momentum/Adam beta1
                    'weight_decay': 5e-4,  # optimizer weight decay
                    'giou': 0.05,  # giou loss gain
                    'cls': 0.5,  # cls loss gain
                    'cls_pw': 1.0,  # cls BCELoss positive_weight
                    'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
                    'obj_pw': 1.0,  # obj BCELoss positive_weight
                    'iou_t': 0.20,  # iou training threshold
                    'anchor_t': 4.0,  # anchor-multiple threshold
                    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
                    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                    'degrees': 0.0,  # image rotation (+/- deg)
                    'translate': 0.0,  # image translation (+/- fraction)
                    'scale': 0.5,  # image scale (+/- gain)
                    'shear': 0.0}  # image shear (+/- deg)

    def load_model(self, hyp, tb_writer, opt, device, epochs):
        self.hyp = hyp
        self.tb_writer = tb_writer
        self.opt = opt
        self.device = device

        # print(f'Hyperparameters {hyp}')
        self.log_dir = self.tb_writer.log_dir if self.tb_writer else 'runs/evolution'  # run directory
        self.wdir = str(Path(self.log_dir) / 'weights') + os.sep  # weights directory
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = self.log_dir + os.sep + 'results.txt'
        self.epochs, batch_size, self.total_batch_size, weights, rank = \
            self.opt.epochs, self.opt.batch_size, self.opt.total_batch_size, self.opt.weights, self.opt.local_rank


        # TODO: Init DDP logging. Only the first process is allowed to log.
        # Since I see lots of print here, the logging configuration is skipped here. We may see repeated outputs.

        # Save run settings
        with open(Path(self.log_dir) / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(Path(self.log_dir) / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)

        # Configure
        init_seeds(2 + rank)
        with open(self.opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        train_path = data_dict['train']
        test_path = data_dict['val']
        self.nc, self.names = (1, ['item']) if opt.single_cls else (
        int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check

        # Remove previous results
        if rank in [-1, 0]:
            for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_file):
                os.remove(f)

        # Create model
        self.model = Model(self.opt.cfg, nc=self.nc).to(self.device)

        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride)
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= batch_size * self.accumulate / self.nbs  # scale weight_decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else

        if self.hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Load Model
        with torch_distributed_zero_first(rank):
            google_utils.attempt_download(weights)
        self.start_epoch, self.best_fitness = 0, 0.0
        if weights.endswith('.pt'):  # pytorch format
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint

            # load model
            try:
                exclude = ['anchor']  # exclude keys
                ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                                 if k in self.model.state_dict() and not any(x in k for x in exclude)
                                 and self.model.state_dict()[k].shape == v.shape}
                self.model.load_state_dict(ckpt['model'], strict=False)
                print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(self.model.state_dict()), weights))
            except KeyError as e:
                s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                    "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                    % (weights, self.opt.cfg, weights, weights)
                raise KeyError(s) from e

            # load optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (self.opt.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

        # Mixed precision training https://github.com/NVIDIA/apex
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # DP mode
        if self.device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            print('\nself.model', self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.device.type != 'cpu' and rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            print('Using SyncBatchNorm()')

        # Exponential moving average
        self.model = attempt_load(weights, map_location=device)  # load FP32 model

        self.ema = torch_utils.ModelEMA(self.model) if rank in [-1, 0] else None

        # DDP mode
        if self.device.type != 'cpu' and rank != -1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        # Trainloader
        self.dataloader, self.dataset = create_dataloader(train_path, self.imgsz, batch_size, self.gs, self.opt,
                                                          hyp=self.hyp, augment=True, cache=self.opt.cache_images,
                                                          rect=self.opt.rect, local_rank=rank,
                                                          world_size=self.opt.world_size)
        mlc = np.concatenate(self.dataset.labels, 0)[:, 0].max()  # max label class
        self.nb = len(self.dataloader)  # number of batches
        assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, self.nc, opt.data, self.nc - 1)

        # Testloader
        self.testloader = create_dataloader(test_path, self.imgsz_test, batch_size, self.gs, self.opt,
                                       hyp=self.hyp, augment=False, cache=self.opt.cache_images, rect=True)[0]

        if rank in [-1, 0]:
            # local_rank is set to -1. Because only the first process is expected to do evaluation.
            self.testloader = create_dataloader(test_path, self.imgsz_test, self.total_batch_size, self.gs, self.opt,
                                                hyp=self.hyp, augment=False, cache=self.opt.cache_images, rect=True,
                                                local_rank=-1, world_size=self.opt.world_size)[0]

        # Model parameters
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device)  # attach class weights
        self.model.names = self.names

        # Class frequency
        if rank in [-1, 0]:
            labels = np.concatenate(self.dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.
            # model._initialize_biases(cf.to(device))
            plot_labels(labels, save_dir=self.log_dir)
            if self.tb_writer:
                # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                self.tb_writer.add_histogram('classes', c, 0)

            # Check anchors
            if not self.opt.noautoanchor:
                check_anchors(self.dataset, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)

        self.nw = max(3 * self.nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.maps = np.zeros(self.nc)  # mAP per class
        self.results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

    def train(self, e):
        t0 = time.time()
        rank = self.opt.local_rank
        if rank in [0, -1]:
            print('Image sizes %g train, %g test' % (self.imgsz, self.imgsz_test))
            print('Using %g dataloader workers' % self.dataloader.num_workers)
            print('Starting training for %g epochs...' % self.epochs)
        # torch.autograd.set_detect_anomaly(True)
        ##########################################################################################################
        # Start training
        for epoch in range(e, e + 1):  # epoch ------------------------------------------------------------------
            self.model.train()

            # Update image weights (optional)
            # When in DDP mode, the generated indices will be broadcasted to synchronize dataset.
            if self.dataset.image_weights:
                # Generate indices.
                if rank in [-1, 0]:
                    w = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2  # class weights
                    image_weights = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=w)
                    self.dataset.indices = random.choices(range(self.dataset.n), weights=image_weights,
                                                     k=self.dataset.n)  # rand weighted idx
                # Broadcast.
                if rank != -1:
                    indices = torch.zeros([self.dataset.n], dtype=torch.int)
                    if rank == 0:
                        indices[:] = torch.from_tensor(self.dataset.indices, dtype=torch.int)
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        self.dataset.indices = indices.cpu().numpy()

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            self.mloss = torch.zeros(4, device=self.device)  # mean losses
            if rank != -1:
                self.dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(self.dataloader)
            if rank in [-1, 0]:
                print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()

            # batch -------------------------------------------------------------
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + self.nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

                # Warmup
                if ni <= self.nw:
                    xi = [0, self.nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, self.hyp['momentum']])

                # Multi-scale
                if self.opt.multi_scale:
                    sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                pred = self.model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(self.device), self.model)  # scaled by batch_size
                if rank != -1:
                    loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return self.results

                # Backward
                if mixed_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Optimize
                if ni % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.update(self.model)

                # Print
                if rank in [-1, 0]:
                    self.mloss = (self.mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    self.s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, self.epochs - 1), mem, * self.mloss, targets.shape[0], imgs.shape[-1])
                    pbar.set_description(self.s)

                    # Plot
                    if ni < 3:
                        f = str(Path(self.log_dir) / ('train_batch%g.jpg' % ni))  # filename
                        result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                        if self.tb_writer and result is not None:
                            self.tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                            # tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # end batch --------------------------------------------------------------------------------------------

            # Scheduler
            self.scheduler.step()

            # Only the first process in DDP mode is allowed to log or save checkpoints.
            if rank in [-1, 0]:
                # mAP
                if self.ema is not None:
                    self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                # final_epoch = epoch + 1 == self.epochs
                final_epoch = True
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    self.results, maps, times = test.test(self.opt.data,
                                                     batch_size=self.total_batch_size,
                                                     imgsz=self.imgsz_test,
                                                     save_json=final_epoch and self.opt.data.endswith(
                                                         os.sep + 'coco.yaml'),
                                                     model=self.ema.ema.module if hasattr(self.ema.ema,
                                                                                     'module') else self.ema.ema,
                                                     single_cls=self.opt.single_cls,
                                                     dataloader=self.testloader,
                                                     save_dir=self.log_dir)

                    # Write
                    with open(self.results_file, 'a') as f:
                        f.write(
                            self.s + '%10.4g' * 7 % self.results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                    if len(self.opt.name) and self.opt.bucket:
                        os.system(
                            'gsutil cp %s gs://%s/results/results%s.txt' % (self.results_file, self.opt.bucket,
                                                                            self.opt.name))

                    # Tensorboard
                    if self.tb_writer:
                        tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                                'metrics/mAP_0.5:0.95',
                                'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                        for x, tag in zip(list(self.mloss[:-1]) + list(self.results), tags):
                            self.tb_writer.add_scalar(tag, x, epoch)

                    # Update best mAP
                    fi = fitness(
                        np.array(self.results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                    if fi > self.best_fitness:
                        self.best_fitness = fi

                # Save model
                save = (not self.opt.nosave) or (final_epoch and not self.opt.evolve)
                if save:
                    with open(self.results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': epoch,
                                'best_fitness': self.best_fitness,
                                'training_results': f.read(),
                                'model': self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema,
                                'optimizer': None if final_epoch else self.optimizer.state_dict()}

                    # Save last, best and delete
                    torch.save(ckpt, self.last)
                    if (self.best_fitness == fi) and not final_epoch:
                        torch.save(ckpt, self.best)
                    del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        if rank in [-1, 0]:
            # Strip optimizers
            n = ('_' if len(self.opt.name) and not self.opt.name.isnumeric() else '') + self.opt.name
            fresults, flast, fbest = 'results%s.txt' % n, self.wdir + 'last%s.pt' % n, self.wdir + 'best%s.pt' % n
            for f1, f2 in zip([self.wdir + 'last.pt', self.wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    ispt = f2.endswith('.pt')  # is *.pt
                    strip_optimizer(f2) if ispt else None  # strip optimizer
                    os.system('gsutil cp %s gs://%s/weights' % (
                        f2, self.opt.bucket)) if self.opt.bucket and ispt else None  # upload
            # Finish
            if not self.opt.evolve:
                plot_results(save_dir=self.log_dir)  # save as results.png
            print('%g epochs completed in %.3f hours.\n' % (epoch - self.start_epoch + 1, (time.time() - t0) / 3600))

        dist.destroy_process_group() if rank not in [-1, 0] else None
        torch.cuda.empty_cache()
        return self.results

    def eval(self, task):
        print('\n detection evaluation start - {} task'.format(task))
        results = test_rl.test(data=self.opt_eval.data, batch_size=self.opt_eval.batch_size, imgsz=self.imgsz_test,
                               conf_thres=self.opt_eval.conf_thres, iou_thres=self.opt_eval.iou_thres,
                               model=self.ema.ema.module if hasattr(self.ema.ema, 'module') else self.ema.ema,
                               task=task)

        return results

    def test(self, inputs, label_path, ind, i):
        results = test.test_rl(inputs, label_path, ind, i, weights=self.opt.weights, device=self.opt.device, batch_size=self.total_batch_size, imgsz=self.imgsz_test,
                                     model=self.ema.ema.module if hasattr(self.ema.ema, 'module') else self.ema.ema,
                                     single_cls=self.opt.single_cls)

        return results

    def main(self, epochs):
        self.last = get_latest_run() if self.opt.resume == 'get_last' else self.opt.resume  # resume from most recent run
        if self.last and not self.opt.weights:
            print(f'Resuming training from {self.last}')
        self.opt.weights = self.last if self.opt.resume and not self.opt.weights else self.opt.weights
        if self.opt.local_rank in [-1, 0]:
            check_git_status()
        self.opt.cfg = check_file(self.opt.cfg)  # check file
        self.opt.data = check_file(self.opt.data)  # check file
        if self.opt.hyp:  # update hyps
            self.opt.hyp = check_file(self.opt.hyp)  # check file
            with open(self.opt.hyp) as f:
                self.hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps
        self.opt.img_size.extend([self.opt.img_size[-1]] * (2 - len(self.opt.img_size)))  # extend to 2 sizes (train, test)
        self.device = torch_utils.select_device(self.opt.device, apex=False, batch_size=self.opt.batch_size)
        self.opt.total_batch_size = self.opt.batch_size
        self.opt.world_size = 1
        if self.device.type == 'cpu':
            mixed_precision = False
        elif self.opt.local_rank != -1:
            # DDP mode
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            self.device = torch.device("cuda", self.opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

            self.opt.world_size = dist.get_world_size()
            assert self.opt.batch_size % self.opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size
        # print(self.opt)

        # Train
        if not self.opt.evolve:
            if self.opt.local_rank in [-1, 0]:
                # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
                self.tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', self.opt.name))
            else:
                self.tb_writer = None
            self.load_model(self.hyp, self.tb_writer, self.opt, self.device, epochs)

        # Evolve hyperparameters (optional)
        else:
            assert self.opt.local_rank == -1, "DDP mode currently not implemented for Evolve!"

            self.tb_writer = None
            self.opt.notest, self.opt.nosave = True, True  # only test/save final epoch
            if self.opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % self.opt.bucket)  # download evolve.txt if exists

            for _ in range(10):  # generations to evolve
                if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.9, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                    ng = len(g)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(self.hyp.keys()):  # plt.hist(v.ravel(), 300)
                        self.hyp[k] = x[i + 7] * v[i]  # mutate

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9),
                          (0, 3)]
                for k, v in zip(keys, limits):
                    self.hyp[k] = np.clip(self.hyp[k], v[0], v[1])

                # Train mutation
                # results = self.load_model(self.device, self.tb_writer, epochs)
                self.load_model(self.hyp, self.tb_writer, self.opt, self.device, epochs)

                # Write mutation results
                # print_mutation(self.hyp, results, self.opt.bucket)

                # Plot results
                # plot_evolution_results(hyp)
