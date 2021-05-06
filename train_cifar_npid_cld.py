'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import datasets
import math
import glob 

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN
from spectral_clustering import spectral_clustering, pairwise_cosine_similarity, KMeans
from lib.lr_scheduler import get_scheduler
from torch.nn.parallel import DistributedDataParallel
from datasets.dataloader import get_dataloader

import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--save-dir', default='checkpoint/', type=str, help='path to save checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str, help='datasets to train')
parser.add_argument('--save-interval', default=100, type=int,
                    help='interval for saving scheckpoint')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')

parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    choices=["step", "cosine"], help="learning rate scheduler")
parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                    help='for step scheduler. decay rate for learning rate')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')

parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size of each iteration')
parser.add_argument('--recompute-memory', action='store_true', help='whether recomputer memory bank')
parser.add_argument('--clusters', default=10, type=int,
                    help='num of clusters for clustering')
parser.add_argument('--k_eigen', default=10, type=int,
                    help='num of eigenvectors for k-way normalized cuts')
parser.add_argument('--cld_t', default=0.07, type=float,
                    help='temperature for clustering')
parser.add_argument('--two-imgs', action='store_true', help='Whether use two randomly processed views')
parser.add_argument('--three-imgs', action='store_true', help='Whether use three randomly processed views')
parser.add_argument('--use-kmeans', action='store_true', help='Whether use k-means for clustering. Use normalized cuts if it is False')
parser.add_argument('--num_iters', default=20, type=int,
                    help='num of iters for clustering')
parser.add_argument('--Lambda', default=1.0, type=float, help='Lambda for groupDis branch')

# misc
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt-level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument("--num_workers", type=int, default=2, help='number of workers for data loader')

args = parser.parse_args()
args.lr = args.batch_size / 128.0 * args.lr
print('INFO CONFIG IS: ', args)

if args.amp:
    try:
        # noinspection PyUnresolvedReferences
        from apex import amp
    except ImportError:
        amp = None

def write_log(args, file_name, epoch, key, top1, top5):
    acc_file = open(os.path.join(args.save_dir, file_name), 'a')
    # Append accuracy to txt file
    acc_file.write('Epoch {} {}: top-1 {:.2f} top5 {:.2f}\n'.format(epoch, key, top1*100., top5*100.))
    # Close the file
    acc_file.close()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

best_acc1 = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Data
print('==> Preparing data..')
trainloader, testloader, ndata = get_dataloader(args)

print('==> Building model..')
net = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len)

# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

net.to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
if device == 'cuda':
    if args.opt_level != "O0":
        if amp is None:
            print("apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                            "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.opt_level = "O0"
        else:
            net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level)
    net = DistributedDataParallel(net, device_ids=[args.local_rank], broadcast_buffers=False)
    cudnn.benchmark = True

scheduler = get_scheduler(optimizer, len(trainloader), args)

# Model
if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_acc1 = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

criterion_cld = nn.CrossEntropyLoss()
criterion_cld.to(device)

lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    sys.exit(0)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    torch.set_num_threads(1)
    if args.lr_scheduler == 'cosine':
        trainloader.sampler.set_epoch(epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    train_CLD_loss = AverageMeter()
    train_CLD_acc = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        targets, indexes = targets.to(device), indexes.to(device)

        # If two_imgs: one is used for F1, another is used for F2. F1 comes from branch1 of net
        # F2 comes from branch1 if only one branch exists else branch2.
        if args.two_imgs:
            inputs1 = inputs[0].to(device)
            inputs2 = inputs[1].to(device)
        else:
            inputs1 = inputs.to(device)
        optimizer.zero_grad()

        features_insDis1, features_batchDis1 = net(inputs1, two_branch=True)
        outputs1 = lemniscate(features_insDis1, indexes)

        # NCE loss
        insDis_loss = criterion(outputs1, indexes)

        if args.two_imgs:
            features_insDis2, features_batchDis2 = net(inputs2, two_branch=True)
            outputs2 = lemniscate(features_insDis2, indexes)
            # NCE loss
            loss_nce_2 = criterion(outputs2, indexes)
            insDis_loss = (insDis_loss + loss_nce_2)/2

        # K-way normalized cuts or k-Means. Default: k-Means
        if args.use_kmeans:
            cluster_label1, centroids1 = KMeans(features_batchDis1, K=args.clusters, Niters=args.num_iters)
            cluster_label2, centroids2 = KMeans(features_batchDis2, K=args.clusters, Niters=args.num_iters)
        else:
            cluster_label1, centroids1 = spectral_clustering(features_batchDis1, K=args.k_eigen,
                        clusters=args.clusters, Niters=args.num_iters)
            cluster_label2, centroids2 = spectral_clustering(features_batchDis2, K=args.k_eigen,
                        clusters=args.clusters, Niters=args.num_iters)

        # instance-group discriminative learning
        affnity1 = torch.mm(features_batchDis1, centroids2.t())
        CLD_loss = criterion_cld(affnity1.div_(args.cld_t), cluster_label2)

        affnity2 = torch.mm(features_batchDis2, centroids1.t())
        CLD_loss = (CLD_loss + criterion_cld(affnity2.div_(args.cld_t), cluster_label1))/2

        # get cluster label prediction accuracy
        _, cluster_pred = torch.topk(affnity1, 1)
        cluster_pred = cluster_pred.t()
        correct = cluster_pred.eq(cluster_label2.view(1, -1).expand_as(cluster_pred))
        correct_all = correct[0].view(-1).float().sum(0).mul_(100.0/inputs1.size(0))
        train_CLD_acc.update(correct_all.item(), inputs1.size(0))

        # total loss
        loss = insDis_loss + args.Lambda*CLD_loss

        if torch.isnan(loss):
            print('INFO loss is nan! Backward skipped')
            continue
        # loss.backward()
        if args.opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item(), inputs1.size(0))
        train_CLD_loss.update(CLD_loss.item(), inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        lr = optimizer.param_groups[0]['lr']
        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}]'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'lr: {:.6f} '
                'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                'CLD loss: {train_cld_loss.val:.4f} ({train_cld_loss.avg:.4f}) '
                'Group acc: {train_CLD_acc.val:.4f} ({train_CLD_acc.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), lr, batch_time=batch_time, 
                data_time=data_time, train_loss=train_loss, train_cld_loss=train_CLD_loss,
                train_CLD_acc=train_CLD_acc))

num_files = glob.glob(args.save_dir + '/' + args.dataset + '_acc_train_cld*')
acc_file_name = args.dataset + '_acc_train_cld' + '_' + 'epochs_200' + '_' + str(len(num_files)) + '.txt'

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)

    if epoch % 1 == 0:
        if args.dataset == 'stl10-full':
            acc1, acc5 = kNN(epoch, net, lemniscate, labeledTrainloader, testloader, 200, args.nce_t, True)
        else:
            acc1, acc5 = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, args.recompute_memory)
    write_log(args, acc_file_name, epoch, key='Acc', top1=acc1, top5=acc5)

    if acc1 > best_acc1 or (epoch+1) % args.save_interval==0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
            'acc': acc1,
            'epoch': epoch,
        }
        if (epoch+1) % args.save_interval == 0:
            file_name = "ckpt_{}_nce_t_{}_nce_k_{}_epoch_{}.t7".format(
                args.dataset, str(args.nce_t), str(args.nce_k), str(epoch+1))
            torch.save(state, os.path.join(args.save_dir,file_name))
        if acc1 > best_acc1:
            file_name = "ckpt_{}_nce_t_{}_nce_k_{}.t7".format(
                args.dataset, str(args.nce_t), str(args.nce_k))
            torch.save(state, os.path.join(args.save_dir,file_name))
            best_acc1 = acc1
            best_acc5 = acc5

    print('best accuracy: {:.2f} {:.2f}'.format(best_acc1*100, best_acc5*100))
    print(args)

if args.dataset == 'stl10-full':
    acc1, acc5 = kNN(epoch, net, lemniscate, labeledTrainloader, testloader, 200, args.nce_t, True)
else:
    acc1, acc5 = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, True)

write_log(args, acc_file_name, epoch, key='Acc-best', top1=best_acc1, top5=best_acc5)

print('last accuracy: {:.2f} {:.2f}'.format(acc1*100, acc5*100))
print('best accuracy: {:.2f} {:.2f}'.format(best_acc1*100, best_acc5*100))
print(args)
