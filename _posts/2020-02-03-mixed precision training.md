---
layout: post
title: ## Mixed precision training
date: 2020-02-03
---

In Deep Learning, computations are usually done in Single Precision(32 bit floats - FP32). Half-precision is 16 bits - FP16.
Mixed Precision is the combined use of different numerical precisions. Mixed Precision training offers significant speedup
by performing operations in half-precision format, while storing minimal information in single-precision.

Tensor Cores on Volta and Turing architectures(e.g. RTX 2080 Ti) provide significant speedups when switching to Mixed 
Precision training. 

Changes needed in the training loop(Pytorch):

1. Imprecise Weight Updates -> 'master' weights in FP32

    * Use half() tensors for the Inputs/Outputs and Model.
    * Store the master copy of model weights in FP32
    * Do forward pass and loss calculation using FP16 model.
    * backpropagate the gradients in FP16.
    * Update the master copy(FP32) using the gradients.
    * Copy the master weights to FP16 model.

2. Gradients may underflow -> Loss(Gradient) scaling

3. Maintain precision for reductions(Loss) -> Accumulate in FP32


References:
-----------
1. [Tensor Cores](https://devblogs.nvidia.com/video-mixed-precision-techniques-tensor-cores-deep-learning)
2. [NVIDIA GTC talk](http://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf)
3. [Fast ai forum thread](https://forums.fast.ai/t/mixed-precision-training/20720)
