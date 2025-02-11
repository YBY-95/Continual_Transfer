import torchvision
from torchvision import transforms, datasets
from .wrapper import CacheClassLabel
import os
from natsort import natsorted
from torch.utils.data import Dataset

def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def ZXJ_GD(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    numerical_folder = []
    class_names = os.listdir(dataroot+r'\\train')
    import re
    for name in class_names:
        if re.match(r'^\d+$', name):
            numerical_folder.append(name)
    class_names = natsorted(numerical_folder)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    # 根据你的实际类别修改

    train_dataset = CustomImageFolder(root=dataroot+r'\\train', custom_class_to_idx=class_to_idx, transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = CustomImageFolder(root=dataroot+r'\\val', custom_class_to_idx=class_to_idx, transform=val_transform)
    val_dataset = CacheClassLabel(val_dataset)


    # train_dataset = datasets.ImageFolder(root=dataroot+r'\\train', transform=train_transform)
    # train_dataset = CacheClassLabel(train_dataset)
    #
    # val_dataset = datasets.ImageFolder(root=dataroot+r'\\val',  transform=val_transform)
    # val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

def numerical_sort(x):
    return int(x[0])


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, custom_class_to_idx=None):
        super(CustomImageFolder, self).__init__(root,
                                                transform=transform,
                                                target_transform=target_transform)
        if custom_class_to_idx is not None:
            self.class_to_idx = custom_class_to_idx
            self.classes = [cls for idx, cls in sorted(self.class_to_idx.items(), key=lambda item: item[1])]

    # def find_classes(self, directory):
    #     # 如果已经提供了custom_class_to_idx，则不需要再次查找类
    #     if hasattr(self, 'class_to_idx'):
    #         return self.classes, self.class_to_idx
    #     return super(CustomImageFolder, self).find_classes(directory)

    # 自定义类别到索引的映射