"""
Code for MoCo pre-training

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
import argparse
import os
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from lib.NCEAverage import MemoryMoCo
from lib.NCECriterion import NCESoftmaxLoss
from utils.logger import setup_logger
import models
from lib.utils import AverageMeter, MyHelpFormatter, DistributedShufle, set_bn_train, moment_update
from lib.lr_scheduler import get_scheduler
from datasets.dataloader import get_dataloader
from test import NN, kNN
import torch.nn as nn
from spectral_clustering import spectral_clustering, pairwise_cosine_similarity, KMeans

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('moco training', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to training')
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'],
                        help="augmentation type: NULL for normal supervised aug, CJ for aug with ColorJitter")
    parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')

    # model and loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce-k', type=int, default=4096, help='num negative sampler')
    parser.add_argument('--nce-t', type=float, default=0.1, help='NCE temperature')
    parser.add_argument('--low-dim', default=128, type=int,
                        metavar='D', help='feature dimension')
    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.03,
                        help='base learning when batch size = 128. final lr is determined by linear scale')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--save-dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    # CLD related arguments
    parser.add_argument('--clusters', default=10, type=int,
                        help='num of clusters for spectral clustering')
    parser.add_argument('--k-eigen', default=10, type=int,
                        help='num of eigenvectors for k-way normalized cuts')
    parser.add_argument('--cld_t', default=0.07, type=float,
                        help='temperature for spectral clustering')
    parser.add_argument('--use-kmeans', action='store_true', help='Whether use k-means for clustering. \
                        Use Normalized Cuts if it is False')
    parser.add_argument('--num-iters', default=20, type=int,
                        help='num of iters for clustering')
    parser.add_argument('--Lambda', default=1.0, type=float,
                        help='weight of mutual information loss')
    parser.add_argument('--two-imgs', action='store_true', help='Whether use two randomly processed views')
    parser.add_argument('--three-imgs', action='store_true', help='Whether use three randomly processed views')
    parser.add_argument('--normlinear', action='store_true', help='whether use normalization linear layer')
    parser.add_argument('--aug-plus', action='store_true', help='whether add strong augmentation')
    parser.add_argument('--erasing', action='store_true', help='whether add random erasing as an augmentation')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args


def build_model(args):
    model = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()
    model_ema = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    return model, model_ema

def load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    args.best_acc = checkpoint['best_acc']
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler, best_acc):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    torch.save(state, os.path.join(args.save_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pth'))


def main(args):
    args.best_acc = 0
    best_acc5 = 0

    # Data
    print('==> Preparing data..')
    train_loader, test_loader, ndata = get_dataloader(args, add_erasing=args.erasing, aug_plus=args.aug_plus)

    logger.info(f"length of training dataset: {ndata}")

    # Model
    model, model_ema = build_model(args)
    contrast = MemoryMoCo(128, args.nce_k, args.nce_t, thresh=0).cuda()
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.batch_size * dist.get_world_size() / 128 * args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.amp_opt_level != "O0":
        if amp is None:
            logger.warning(f"apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
            model_ema = amp.initialize(model_ema, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.save_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.lr_scheduler == 'cosine':
            train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args)

        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('ins_prob', prob, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if args.dataset == 'stl10-full':
            acc, acc5 = kNN(epoch, model, contrast, labeledTrainloader, test_loader, 200, args.nce_t, True)
        else:
            acc, acc5 = kNN(epoch, model, contrast, train_loader, test_loader, 200, args.nce_t, True)
        if acc >= args.best_acc: 
            args.best_acc = acc
            best_acc5 = acc5
        logger.info('KNN top-1 precion: {:.4f} {:.4f}, best is: {:.4f} {:.4f}'.format(acc*100., \
            acc5*100., args.best_acc*100., best_acc5*100))
        logger.info(str(args))

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler, args.best_acc)
    if args.dataset == 'stl10-full':
        acc1, acc5 = kNN(epoch, model, contrast, labeledTrainloader, test_loader, 200, args.nce_t, True)
    else:
        acc1, acc5 = kNN(epoch, model, contrast, train_loader, test_loader, 200, args.nce_t, True)

    logger.info('KNN top-1 and top-5 precion with recomputed memory bank: {:.4f} {:.4f}'.format(acc1*100., acc5*100))
    logger.info('Best KNN top-1 and top-5 precion: {:.4f} {:.4f}'.format(args.best_acc*100., best_acc5*100))
    logger.info(str(args))


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args):
    """
    one epoch training for moco
    """
    model.train()
    set_bn_train(model_ema)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    train_CLD_loss = AverageMeter()
    train_CLD_acc = AverageMeter()
    criterion_cld = nn.CrossEntropyLoss().cuda()

    end = time.time()
    torch.set_num_threads(1)
    for idx, ((x1, x2, x3), targets, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = x1.size(0)

        x1 = x1.cuda()
        x2 = x2.cuda()
        x3 = x3.cuda()

        feat_q1, features_groupDis1 = model(x1, two_branch=True)
        feat_q2, features_groupDis2 = model(x2, two_branch=True)

        with torch.no_grad():
            x3_shuffled, backward_inds = DistributedShufle.forward_shuffle(x3, epoch)
            feat_k3, features_groupDis3 = model_ema(x3_shuffled, two_branch=True)
            feat_k_all, feat_k3, features_groupDis3 = DistributedShufle.backward_shuffle(
                feat_k3, backward_inds, return_local=True, branch_two=features_groupDis3)

        # NCE loss
        out = contrast(feat_q1, feat_k3, feat_k_all, update=False)
        loss_1 = criterion(out)

        out = contrast(feat_q2, feat_k3, feat_k_all, update=True)
        loss_2 = criterion(out)
        loss = (loss_1 + loss_2)/2

        prob = F.softmax(out, dim=1)[:, 0].mean()

        # K-way normalized cuts or k-Means. Default: k-Means
        if args.use_kmeans:
            cluster_label1, centroids1 = KMeans(features_groupDis1, K=args.clusters, Niters=args.num_iters)
            cluster_label2, centroids2 = KMeans(features_groupDis2, K=args.clusters, Niters=args.num_iters)
        else:
            cluster_label1, centroids1 = spectral_clustering(features_groupDis1, K=args.k_eigen,
                        clusters=args.clusters, Niters=args.num_iters)
            cluster_label2, centroids2 = spectral_clustering(features_groupDis2, K=args.k_eigen,
                        clusters=args.clusters, Niters=args.num_iters)

        # instance-group discriminative learning
        affnity1 = torch.mm(features_groupDis1, centroids2.t())
        CLD_loss = criterion_cld(affnity1.div_(args.cld_t), cluster_label2)

        affnity2 = torch.mm(features_groupDis2, centroids1.t())
        CLD_loss = (CLD_loss + criterion_cld(affnity2.div_(args.cld_t), cluster_label1))/2

        # get cluster label prediction accuracy
        _, cluster_pred = torch.topk(affnity1, 1)
        cluster_pred = cluster_pred.t()
        correct = cluster_pred.eq(cluster_label2.view(1, -1).expand_as(cluster_pred))
        correct_all = correct[0].view(-1).float().sum(0).mul_(100.0/x1.size(0))
        train_CLD_acc.update(correct_all.item(), x1.size(0))

        # total loss
        loss = loss + args.Lambda*CLD_loss

        if torch.isnan(loss):
            print('INFO loss is nan! Backward skipped')
            continue

        # backward
        optimizer.zero_grad()
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha)
        train_CLD_loss.update(CLD_loss.item(), x1.size(0))

        # update meters
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        lr = optimizer.param_groups[0]['lr']
        if idx % args.print_freq == 0:
            logger.info(f'Train: [{epoch}][{idx}/{len(train_loader)}] lr: {lr:.5f}\t'
                        f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                        f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})\t'
                        f'CLD loss {train_CLD_loss.val:.3f} ({train_CLD_loss.avg:.3f})\t'
                        f'Top-1 acc {train_CLD_acc.val:.3f} ({train_CLD_acc.avg:.3f})')

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.save_dir, exist_ok=True)
    logger = setup_logger(output=opt.save_dir, distributed_rank=dist.get_rank(), name="moco+cld")
    if dist.get_rank() == 0:
        path = os.path.join(opt.save_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
