import torch
from models.resnet import Reduced_ResNet18
from torchvision import transforms
import torch.nn as nn


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'coil100': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
}


n_classes = {
    'cifar100': 100,
    'coil100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
}


transforms_match = {
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'coil100': transforms.Compose([
            transforms.ToTensor()])
}


def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.data == 'cifar100':
        return Reduced_ResNet18(nclass)
    elif params.data == 'cifar10':
        return Reduced_ResNet18(nclass)
    elif params.data == 'coil100':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(2560, nclass, bias=True)
        return model
    elif params.data == 'mini_imagenet':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        return model


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        print(f"LR: {lr}, optimizer: {optimizer}, wd: {wd}")
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
