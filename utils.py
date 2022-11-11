import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

# knowledge distillation loss function
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""


    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


def moving_average(net1, net2, alpha=1, targets=None):
    layer_list_map = {1: l1, 2: l2, 3: l3, 4: l4}
    layer_list = []
    for i in targets:
        layer_list += layer_list_map[i]
    for (name, param1), (_, param2) in zip(net1.named_parameters(), net2.named_parameters()):
        if name in layer_list:
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for batch in loader:
        input = batch['input']
        input = input.to(device)
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def get_layers(model):
    l0, l1, l2, l3, l4 = [], [], [], [], []
    if model == "PreActResNet18":
        l1 = ["layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.conv1.weight", "layer1.0.bn2.weight",
              "layer1.0.bn2.bias", "layer1.0.conv2.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias",
              "layer1.1.conv1.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.conv2.weight"]
        l2 = ["layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.conv1.weight", "layer2.0.bn2.weight",
              "layer2.0.bn2.bias", "layer2.0.conv2.weight", "layer2.0.shortcut.0.weight",
              "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.conv1.weight", "layer2.1.bn2.weight",
              "layer2.1.bn2.bias", "layer2.1.conv2.weight"]
        l3 = ["layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.conv1.weight", "layer3.0.bn2.weight",
              "layer3.0.bn2.bias", "layer3.0.conv2.weight", "layer3.0.shortcut.0.weight",
              "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.conv1.weight", "layer3.1.bn2.weight",
              "layer3.1.bn2.bias", "layer3.1.conv2.weight"]
        l4 = ["layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.conv1.weight", "layer4.0.bn2.weight",
              "layer4.0.bn2.bias", "layer4.0.conv2.weight", "layer4.0.shortcut.0.weight",
              "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.conv1.weight", "layer4.1.bn2.weight",
              "layer4.1.bn2.bias", "layer4.1.conv2.weight"]
    # 4 layers break down
    elif model == "WideResNet":
        l1 = ['conv1.weight', 'block1.layer.0.bn1.weight', 'block1.layer.0.bn1.bias', 'block1.layer.0.conv1.weight',
              'block1.layer.0.bn2.weight', 'block1.layer.0.bn2.bias', 'block1.layer.0.conv2.weight',
              'block1.layer.0.convShortcut.weight', 'block1.layer.1.bn1.weight', 'block1.layer.1.bn1.bias',
              'block1.layer.1.conv1.weight', 'block1.layer.1.bn2.weight', 'block1.layer.1.bn2.bias',
              'block1.layer.1.conv2.weight', 'block1.layer.2.bn1.weight', 'block1.layer.2.bn1.bias',
              'block1.layer.2.conv1.weight', 'block1.layer.2.bn2.weight', 'block1.layer.2.bn2.bias',
              'block1.layer.2.conv2.weight', 'block1.layer.3.bn1.weight', 'block1.layer.3.bn1.bias',
              'block1.layer.3.conv1.weight', 'block1.layer.3.bn2.weight', 'block1.layer.3.bn2.bias',
              ]
        l2 = ['block1.layer.3.conv2.weight', 'block1.layer.4.bn1.weight', 'block1.layer.4.bn1.bias',
              'block1.layer.4.conv1.weight', 'block1.layer.4.bn2.weight', 'block1.layer.4.bn2.bias',
              'block1.layer.4.conv2.weight', 'block2.layer.0.bn1.weight', 'block2.layer.0.bn1.bias',
              'block2.layer.0.conv1.weight', 'block2.layer.0.bn2.weight', 'block2.layer.0.bn2.bias',
              'block2.layer.0.conv2.weight', 'block2.layer.0.convShortcut.weight',
              'block2.layer.1.bn1.weight', 'block2.layer.1.bn1.bias', 'block2.layer.1.conv1.weight',
              'block2.layer.1.bn2.weight', 'block2.layer.1.bn2.bias', 'block2.layer.1.conv2.weight',
              'block2.layer.2.bn1.weight', 'block2.layer.2.bn1.bias', 'block2.layer.2.conv1.weight',
              'block2.layer.2.bn2.weight', 'block2.layer.2.bn2.bias']
        l3 = ['block2.layer.2.conv2.weight', 'block2.layer.3.bn1.weight', 'block2.layer.3.bn1.bias',
              'block2.layer.3.conv1.weight', 'block2.layer.3.bn2.weight', 'block2.layer.3.bn2.bias',
              'block2.layer.3.conv2.weight', 'block2.layer.4.bn1.weight', 'block2.layer.4.bn1.bias',
              'block2.layer.4.conv1.weight', 'block2.layer.4.bn2.weight', 'block2.layer.4.bn2.bias',
              'block2.layer.4.conv2.weight', 'block3.layer.0.bn1.weight',
              'block3.layer.0.bn1.bias', 'block3.layer.0.conv1.weight', 'block3.layer.0.bn2.weight',
              'block3.layer.0.bn2.bias', 'block3.layer.0.conv2.weight', 'block3.layer.0.convShortcut.weight',
              'block3.layer.1.bn1.weight', 'block3.layer.1.bn1.bias']
        l4 = ['block3.layer.1.conv1.weight', 'block3.layer.1.bn2.weight', 'block3.layer.1.bn2.bias',
              'block3.layer.1.conv2.weight','block3.layer.2.bn1.weight', 'block3.layer.2.bn1.bias',
              'block3.layer.2.conv1.weight', 'block3.layer.2.bn2.weight',  'block3.layer.2.bn2.bias',
              'block3.layer.2.conv2.weight', 'block3.layer.3.bn1.weight', 'block3.layer.3.bn1.bias',
              'block3.layer.3.conv1.weight','block3.layer.3.bn2.weight', 'block3.layer.3.bn2.bias',
              'block3.layer.3.conv2.weight', 'block3.layer.4.bn1.weight', 'block3.layer.4.bn1.bias',
              'block3.layer.4.conv1.weight', 'block3.layer.4.bn2.weight', 'block3.layer.4.bn2.bias',
              'block3.layer.4.conv2.weight', 'bn1.weight', 'bn1.bias', 'fc.weight', 'fc.bias']


    # 3 layers break down
    # elif model == "WideResNet":
    #     l1 = ['block1.layer.0.bn1.weight', 'block1.layer.0.bn1.bias', 'block1.layer.0.conv1.weight',
    #      'block1.layer.0.bn2.weight', 'block1.layer.0.bn2.bias', 'block1.layer.0.conv2.weight',
    #      'block1.layer.0.convShortcut.weight', 'block1.layer.1.bn1.weight', 'block1.layer.1.bn1.bias',
    #      'block1.layer.1.conv1.weight', 'block1.layer.1.bn2.weight', 'block1.layer.1.bn2.bias',
    #      'block1.layer.1.conv2.weight', 'block1.layer.2.bn1.weight', 'block1.layer.2.bn1.bias',
    #      'block1.layer.2.conv1.weight', 'block1.layer.2.bn2.weight', 'block1.layer.2.bn2.bias',
    #      'block1.layer.2.conv2.weight', 'block1.layer.3.bn1.weight', 'block1.layer.3.bn1.bias',
    #      'block1.layer.3.conv1.weight', 'block1.layer.3.bn2.weight', 'block1.layer.3.bn2.bias',
    #      'block1.layer.3.conv2.weight', 'block1.layer.4.bn1.weight', 'block1.layer.4.bn1.bias',
    #      'block1.layer.4.conv1.weight', 'block1.layer.4.bn2.weight', 'block1.layer.4.bn2.bias', 'block1.layer.4.conv2.weight']
    #     l2 = ['block2.layer.0.bn1.weight', 'block2.layer.0.bn1.bias', 'block2.layer.0.conv1.weight', 'block2.layer.0.bn2.weight',
    #      'block2.layer.0.bn2.bias', 'block2.layer.0.conv2.weight', 'block2.layer.0.convShortcut.weight',
    #      'block2.layer.1.bn1.weight', 'block2.layer.1.bn1.bias', 'block2.layer.1.conv1.weight',
    #      'block2.layer.1.bn2.weight', 'block2.layer.1.bn2.bias', 'block2.layer.1.conv2.weight',
    #      'block2.layer.2.bn1.weight', 'block2.layer.2.bn1.bias', 'block2.layer.2.conv1.weight',
    #      'block2.layer.2.bn2.weight', 'block2.layer.2.bn2.bias', 'block2.layer.2.conv2.weight',
    #      'block2.layer.3.bn1.weight', 'block2.layer.3.bn1.bias', 'block2.layer.3.conv1.weight',
    #      'block2.layer.3.bn2.weight', 'block2.layer.3.bn2.bias', 'block2.layer.3.conv2.weight',
    #      'block2.layer.4.bn1.weight', 'block2.layer.4.bn1.bias', 'block2.layer.4.conv1.weight',
    #      'block2.layer.4.bn2.weight', 'block2.layer.4.bn2.bias', 'block2.layer.4.conv2.weight']
    #     l3 = ['block3.layer.0.bn1.weight',
    #      'block3.layer.0.bn1.bias', 'block3.layer.0.conv1.weight', 'block3.layer.0.bn2.weight',
    #      'block3.layer.0.bn2.bias', 'block3.layer.0.conv2.weight', 'block3.layer.0.convShortcut.weight',
    #      'block3.layer.1.bn1.weight', 'block3.layer.1.bn1.bias', 'block3.layer.1.conv1.weight',
    #      'block3.layer.1.bn2.weight', 'block3.layer.1.bn2.bias', 'block3.layer.1.conv2.weight',
    #      'block3.layer.2.bn1.weight', 'block3.layer.2.bn1.bias', 'block3.layer.2.conv1.weight',
    #      'block3.layer.2.bn2.weight', 'block3.layer.2.bn2.bias', 'block3.layer.2.conv2.weight',
    #      'block3.layer.3.bn1.weight', 'block3.layer.3.bn1.bias', 'block3.layer.3.conv1.weight',
    #      'block3.layer.3.bn2.weight', 'block3.layer.3.bn2.bias', 'block3.layer.3.conv2.weight',
    #      'block3.layer.4.bn1.weight', 'block3.layer.4.bn1.bias', 'block3.layer.4.conv1.weight',
    #      'block3.layer.4.bn2.weight', 'block3.layer.4.bn2.bias', 'block3.layer.4.conv2.weight']
    #     l4 = []

    # 5 layers inside a block break down
    # elif model == "WideResNet":
    #     l0 = ['block1.layer.0.bn1.weight', 'block1.layer.0.bn1.bias', 'block1.layer.0.conv1.weight',
    #           'block1.layer.0.bn2.weight', 'block1.layer.0.bn2.bias', 'block1.layer.0.conv2.weight',
    #           'block1.layer.0.convShortcut.weight', 'block2.layer.0.bn1.weight', 'block2.layer.0.bn1.bias',
    #           'block2.layer.0.conv1.weight', 'block2.layer.0.bn2.weight', 'block2.layer.0.bn2.bias',
    #           'block2.layer.0.conv2.weight', 'block2.layer.0.convShortcut.weight', 'block3.layer.0.bn1.weight',
    #           'block3.layer.0.bn1.bias', 'block3.layer.0.conv1.weight', 'block3.layer.0.bn2.weight',
    #           'block3.layer.0.bn2.bias', 'block3.layer.0.conv2.weight', 'block3.layer.0.convShortcut.weight']
    #
    #     l1 = ['block1.layer.1.bn1.weight', 'block1.layer.1.bn1.bias', 'block1.layer.1.conv1.weight',
    #           'block1.layer.1.bn2.weight', 'block1.layer.1.bn2.bias', 'block1.layer.1.conv2.weight',
    #           'block2.layer.1.bn1.weight', 'block2.layer.1.bn1.bias', 'block2.layer.1.conv1.weight',
    #           'block2.layer.1.bn2.weight', 'block2.layer.1.bn2.bias', 'block2.layer.1.conv2.weight',
    #           'block3.layer.1.bn1.weight', 'block3.layer.1.bn1.bias', 'block3.layer.1.conv1.weight',
    #           'block3.layer.1.bn2.weight', 'block3.layer.1.bn2.bias', 'block3.layer.1.conv2.weight',
    #           ]
    #
    #     l2 = ['block1.layer.2.bn1.weight', 'block1.layer.2.bn1.bias', 'block1.layer.2.conv1.weight',
    #           'block1.layer.2.bn2.weight', 'block1.layer.2.bn2.bias', 'block1.layer.2.conv2.weight',
    #           'block2.layer.2.bn1.weight', 'block2.layer.2.bn1.bias', 'block2.layer.2.conv1.weight',
    #           'block2.layer.2.bn2.weight', 'block2.layer.2.bn2.bias', 'block2.layer.2.conv2.weight',
    #           'block3.layer.2.bn1.weight', 'block3.layer.2.bn1.bias', 'block3.layer.2.conv1.weight',
    #           'block3.layer.2.bn2.weight', 'block3.layer.2.bn2.bias', 'block3.layer.2.conv2.weight']
    #
    #     l3 = ['block1.layer.3.bn1.weight', 'block1.layer.3.bn1.bias', 'block1.layer.3.conv1.weight',
    #           'block1.layer.3.bn2.weight', 'block1.layer.3.bn2.bias', 'block1.layer.3.conv2.weight',
    #           'block2.layer.3.bn1.weight', 'block2.layer.3.bn1.bias', 'block2.layer.3.conv1.weight',
    #           'block2.layer.3.bn2.weight', 'block2.layer.3.bn2.bias', 'block2.layer.3.conv2.weight',
    #           'block3.layer.3.bn1.weight', 'block3.layer.3.bn1.bias', 'block3.layer.3.conv1.weight',
    #           'block3.layer.3.bn2.weight', 'block3.layer.3.bn2.bias', 'block3.layer.3.conv2.weight']
    #
    #     l4 = ['block1.layer.4.bn1.weight', 'block1.layer.4.bn1.bias', 'block1.layer.4.conv1.weight',
    #           'block1.layer.4.bn2.weight', 'block1.layer.4.bn2.bias', 'block1.layer.4.conv2.weight',
    #           'block2.layer.4.bn1.weight', 'block2.layer.4.bn1.bias', 'block2.layer.4.conv1.weight',
    #           'block2.layer.4.bn2.weight', 'block2.layer.4.bn2.bias', 'block2.layer.4.conv2.weight',
    #           'block3.layer.4.bn1.weight', 'block3.layer.4.bn1.bias', 'block3.layer.4.conv1.weight',
    #           'block3.layer.4.bn2.weight', 'block3.layer.4.bn2.bias', 'block3.layer.4.conv2.weight']

    return l0, l1, l2, l3, l4