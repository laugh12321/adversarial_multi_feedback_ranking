# Multi Channel Adversarial Personalized Ranking

## Environment

Python 3.6.8

TensorFlow 1.13.1

Numpy 1.14.6

Pandas 0.24.2

## Quick Start

This command shows the effect of MPR by adding adversarial perturbation on MPR model for dataset yelp in epoch 500 (--adv_epoch). The first 500 epochs are MPR, followed by adversarial training MPR.

```shell
python main.py --dataset ml-1m --adv_epoch 500 --epochs 1000 --eps 0.5 --reg_adv 1 --ckpt 1 --verbose 10 --beta 1 --sampling 'uniform' 
```
