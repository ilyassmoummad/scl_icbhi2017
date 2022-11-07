# Supervised Contrastive Learning for Respiratory Sound Classification
-----
This is the official pytorch implementation of our work [Supervised Contrastive Learning for Respiratory Sound Classification](https://arxiv.org/abs/2210.16192)

## Dependencies:
Launch : ```pip install -r requirements.txt```

## Dataset:
Put the data files in the data folder

## Pretrained models:
Put the pretrained pth files in the panns folder

## Metadata:
```metadata.py``` creates a metadata file in the data folder

## Training:
Launch : ```python3 main.py --method METHOD --bs BS``` \
Replace **METHOD** by the desired training method (**sl** for cross entropy, **scl** for supervised contrastive, or **hybrid** for a combination of both), and **BS** with the desired batch size (128 being the default value)

## Quantitative Results
We optimized hyperparameters for CNN6, and we simply report CNN10 from scratch and pretrained CNN14 scores on ICBHI without any hyperparameter tuning:

| Backbone | Method |     _Sp_    |     _Se_    |     _Sc_    | # of Params | Ext. Dataset |
|:--------:|:------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|   Cnn6   |   CE   | 76.72(3.97) | 31.12(3.72) | 53.92(0.71) |     4.3     |       -      |
|          |   SCL  | 76.17(3.84) | 27.97(3.92) | 52.08(1.06) |             |              |
|          | Hybrid | 75.35(5.47) | 33.84(5.67) |  54.74(0.5) |             |              |
|   Cnn10  |   CE   |  73.45(6.7) |  36.8(6.61) | 55.13(1.56) |     4.8     |       -      |
|          |   SCL  | 74.78(5.89) |  30.38(5.5) | 52.59(1.35) |             |              |
|          | Hybrid | 78.12(6.14) | 33.07(5.43) |  55.6(1.13) |             |              |
|   Cnn6   |   CE   | 70.09(3.08) | 40.39(2.97) | 55.24(0.43) |     4.3     |   AudioSet   |
|          |   SCL  | 75.95(2.31) | 39.15(1.89) | 57.55(0.81) |             |              |
|          | Hybrid | 70.47(2.07) | 43.29(1.83) | 56.89(0.55) |             |              |
|   Cnn14  |   CE   | 75.63(4.13) | 38.13(5.07) | 57.32(0.54) |     75.4    |   AudioSet   |
|          |   SCL  |  80.67(4.2) | 32.93(4.37) |  56.92(0.9) |             |              |
|          | Hybrid | 80.73(3.86) | 34.96(3.59) | 57.85(0.48) |             |              |


## To cite this work:
```
@misc{scl_icbhi2017,
      title={Supervised Contrastive Learning for Respiratory Sound Classification}, 
      author={Ilyass Moummad and Nicolas Farrugia},
      year={2022},
      eprint={2210.16192},
      archivePrefix={arXiv},
      primaryClass={cs.SD}}
```