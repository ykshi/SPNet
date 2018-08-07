# SPNet

This is implementation of TMM17 paper, XXXXXX. In this work, we proposed a efficient and structure-preserving image super-resolution framework by incorporating light-weight architecture and contextualized learning. 

# Prerequisites
- Computer with Linux
- Pytorch 0.3.0
- A NVIDIA GPU with CUDA8.0 installed

# Data Generation

We put the General-100 and Set14 at `./Train`. First, we should run scripts `generate_train.m` and `generate_test.m` to generate sub-images.

# Evaluation
In this implemention, we evaluate SPNet on Set14. In addition, we also provide a baseline model(e.g. FSRCNN) for better comparison. Both of us were trained on General-100 with 1000 epoches. The training code are `main_spnet.py` and `main_cnn_baseline.py`, respectively.

The proposed model achieve well balance beween efficiency and performance. Runing the evaluation script with `sh eval.sh`, with the output as:
```
=========SPNet==========
The testing time is 0.837336 second
Avg. PSNR: 28.7753 dB   Bilinear 27.1091 dB 
========================

=========Baseline==========
The testing time is 0.920399 second
Avg. PSNR: 28.5425 dB   Bilinear 27.1091 dB 
========================
```

Since the boundary contextualized model requires lots of manual efforts and the training process is too complex to provide one-step script. Thus, we provide a model trained with VOC2012 and boundary map in `./checkpoints/main_spnetmodel_pre_trained.pth`. You can run script `sh eval_pre_trained.sh` to re-produce our results with output as: 
```
=========SPNet==========
The testing time is 0.814436 second
Avg. PSNR: 29.2629 dB   Bilinear 27.1091 dB 
========================
```
# Train
We have organized the training code for RCN and BCN components. You can train the model by using the following command:
```
python main_spnet.py
```

# Feedback and Citation
If SPNet helps in your research, you can cite our paper:
```
@article{Shi2017Structure,
  title={Structure-Preserving Image Super-resolution via Contextualized Multi-task Learning},
  author={Shi, Yukai and Wang, Keze and Chen, Chongyu and Xu, Li and Lin, Liang},
  journal={IEEE Transactions on Multimedia},
  volume={PP},
  number={99},
  pages={1-1},
  year={2017},
}
```
Also, if you have any question, please feel free to contact me by sending mail to `shiyk3ATmail2.sysu.edu.cn`.

# Acknowledgement 
This code is heavily rely on [pytorch examples](https://github.com/pytorch/examples), thanks for their great work
