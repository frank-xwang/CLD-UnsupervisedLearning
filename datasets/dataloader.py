import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import datasets

from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_dataloader(args, add_erasing=False, aug_plus=False):
    if 'cifar' in args.dataset or 'kitchen' in args.dataset:
        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            transform_train_list = [
                transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        else:
            transform_train_list = [
                transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        if add_erasing:
            transform_train_list.append(transforms.RandomErasing(p=1.0))
        transform_train = transforms.Compose(transform_train_list)

        if 'kitchen' in args.dataset:
            transform_test = transforms.Compose([
                transforms.Resize((32,32), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    elif 'stl' in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=96, scale=(0.2,1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10Instance(root='./data/CIFAR-10', train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.CIFAR10Instance(root='./data/CIFAR-10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100Instance(root='./data/CIFAR-100', train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.CIFAR100Instance(root='./data/CIFAR-100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'stl10':
        trainset = datasets.STL10(root='./data/STL10', split='train', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.STL10(root='./data/STL10', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = trainset.__len__()

    elif args.dataset == 'stl10-full':
        trainset = datasets.STL10(root='./data/STL10', split='train+unlabeled', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            pin_memory=False, sampler=train_sampler)

        labeledTrainset = datasets.STL10(root='./data/STL10', split='train', download=True, transform=transform_train, two_imgs=args.two_imgs)
        labeledTrain_sampler = torch.utils.data.distributed.DistributedSampler(labeledTrainset)
        labeledTrainloader = torch.utils.data.DataLoader(labeledTrainset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=False, sampler=labeledTrain_sampler)
        testset = datasets.STL10(root='./data/STL10', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = labeledTrainset.__len__()

    elif args.dataset == 'kitchen':
        trainset = datasets.CIFARImageFolder(root='./data/Kitchen-HC/train', train=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        testset = datasets.CIFARImageFolder(root='./data/Kitchen-HC/test', train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()
    
    return trainloader, testloader, ndata