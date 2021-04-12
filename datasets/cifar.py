from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, root='./data/cifar10', train=True, download=True, transform=None, two_imgs=False, three_imgs=False):
        super(CIFAR10Instance, self).__init__(root=root, train=train, download=download, transform=transform)
        self.two_imgs = two_imgs
        self.three_imgs = three_imgs
    
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_imgs:
            img2 = self.transform(img)
            return (img1, img2), target, index
        elif self.three_imgs:
            img2 = self.transform(img)
            img3 = self.transform(img)
            return (img1, img2, img3), target, index
        else:
            return img1, target, index

class CIFAR100Instance(CIFAR10Instance):
    """CIFAR100Instance Dataset.

    This is a subclass of the `CIFAR10Instance` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class CIFARImageFolder(datasets.ImageFolder):
    """CIFAR10Instance Dataset.
    """

    def __init__(self, root='./data/cifar10_LT', train=True, transform=None, two_imgs=False):
        super(CIFARImageFolder, self).__init__(root=root, transform=transform)
        self.two_imgs = two_imgs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        try:
            img_ = self.loader(path)
        except:
            print(path)
        if self.transform is not None:
            img = self.transform(img_)
            if self.two_imgs:
                img2 = self.transform(img_)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.two_imgs:
            return (img, img2), target, index
        else:
            return img, target, index