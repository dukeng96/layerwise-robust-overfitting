# Layer Wise Analysis of Robust Overfitting in Adversarial Training


## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.2.0
- torchvision = 0.4.0

## What is in this repository
Codes for our AT_fixlr_[*layers] and AT_wp_[*layers] methods are avaiable.
- Codes for our fixing learning rate experiments for CIFAR-10, CIFAR-100, and SVHN are in `train_cifar10_fixlr.py`, `train_cifar100_fixlr.py` and `train_svhn_fixlr.py` respectively.
- Codes for our weight pertubation experiments for CIFAR-10, CIFAR-100, and SVHN are in `train_cifar10_wp.py`, `train_cifar100_wp.py` and `train_svhn_wp.py` respectively.

Auto attack codes are in `auto_attacks`
- The codes for robustness evaluation on AA are in `eval.py`.

## How to use it

For AT_fixlr_[*layers] with a PreAct ResNet-18 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python train_cifar10_fixlr.py --target-layers 3 4
``` 
You can configure the target layers as needed, e.g., "1", "1 2", "1 2 3", "2 3 4", etc.

For AT_wp_[*layers] with a PreAct ResNet-18 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python train_cifar10_wp.py --target-layers 3 4
``` 
You can configure the target layers as needed, e.g., "1", "1 2", "1 2 3", "2 3 4", etc.


## Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting

[2] TRADES: https://github.com/yaodongyu/TRADES/

[3] RST: https://github.com/yaircarmon/semisup-adv

[4] AWP: https://github.com/csdongxian/AWP

[5] AutoAttack: https://github.com/fra31/auto-attack