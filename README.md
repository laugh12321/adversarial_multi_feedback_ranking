# Multi-feedback Pairwise Ranking via Adversarial Training for Recommender

This is our official implementation for the paper:

>[WANG Jianfang, FU Zhiyuan, NIU Mingxin, ZHANG Pengbo, ZHANG Qiuling. <b>Multi-feedback Pairwise Ranking via Adversarial Training for Recommender[J]</b>. Chinese Journal of Electronics, 2020, 29(4): 615-622.](http://www.ejournal.org.cn/Jweb_cje/EN/10.1049/cje.2020.05.004)

If you use the codes, please cite our paper . Thanks!

## Introduction

Multi-feedback pairwise ranking method via Adversarial training (AT-MPR) for recommender to enhance the robustness and overall performance in the event of rating pollution.

## Environment Requirement

The code has been tested running under Python 3.6 The required packages are as follows:

- TensorFlow 1.13

- Numpy 1.14

- Pandas 0.24

## Example to Run the Codes

This command shows the effect of MPR by adding adversarial perturbation on MPR model for dataset ml-1m in epoch 500 (--adv_epoch). The first 500 epochs are MPR, followed by adversarial training MPR.

```shell
python AT-MPR.py --dataset ml-1m --adv_epoch 500 --epochs 1000 --eps 0.5 --reg_adv 1 --ckpt 1 --verbose 10 --beta 1 --sampling 'uniform' 
```

Some important arguments:

<table><tr>
<td><img src=https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[eps]%20HR.png border=0></td>
<td><img src=https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/AT-MPR/imgs/[eps]%20NDCG.png border=0></td>
</tr></table>


- `eps`: Used to adjust the intensity of the confrontation, experiments show that the effect is best at `0.5` (see figure, above).

- `beta`: The proportion of implicit feedback in data is the best when `1` is realized.

- `sampling`: Provide two different sampling methods `non-uniform` , `uniform` among which `uniform` performs best in `MovieLens`. 

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

## Dataset

We provide three processed datasets: Yelp(yelp), MovieLens 1 Million (ml-1m) and Ciao (CiaoDVD) in Data

<b>train.rating:</b>

- Train file.
- Each Line is a training instance: userID,  itemID, rating

<b>test.rating:</b>

- Test file.
- Each Line is a testing instance: userID,  itemID, rating


Update: Jul 18, 2020
