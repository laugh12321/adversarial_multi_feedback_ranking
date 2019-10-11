# Multi-feedback Adversarial Personalized Ranking


## Introduct
(AT-MPR) is a new ecommendation framework based on MPR.

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[ML-1M]AT-MPR%20VS%20MPR.png)
![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[CiaoDVD]AT-MPR%20VS%20MPR.png)
![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[Yelp]AT-MPR%20VS%20MPR.png)

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- TensorFlow 1.13.1

- Numpy 1.14.6

- Pandas 0.24.2

## Quick Start

This command shows the effect of MPR by adding adversarial perturbation on MPR model for dataset yelp in epoch 500 (--adv_epoch). The first 500 epochs are MPR, followed by adversarial training MPR.

```shell
python AT-MPR.py --dataset ml-1m --adv_epoch 500 --epochs 1000 --eps 0.5 --reg_adv 1 --ckpt 1 --verbose 10 --beta 1 --sampling 'uniform' 
```



Some important arguments:

<table><tr>
<td><img src=https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[eps]%20HR.png border=0></td>
<td><img src=https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[eps]%20NDCG.png border=0></td>
</tr></table>


- `eps`: 用来调成对抗的强度，实验表明在 `0.5` 时效果最佳 (上图).

- `beta`: 数据中隐式反馈所占的比例，实现表明 `1` 时效果最佳.

- `sampling`: 提供两种不同的采样方式 `non-uniform`， `uniform` 其中 `uniform` 在 `MovieLens` 中表现最好 

<b>More Details:</b>

Use python main.py -h to get more argument setting details.

```
-h, --help            show this help message and exit
--path [PATH]         Input data path.
--dataset [DATASET]   Choose a dataset.
--verbose VERBOSE     Evaluate per X epochs.
--epochs EPOCHS       Number of epochs.
--adv_epochs          The epoch # that starts adversarial training (before that are normal MPR training). 
......
```
