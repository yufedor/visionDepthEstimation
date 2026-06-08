# Project 5 — Lightweight Diffusion Models

## Accelerating Training / Inference for Resource-Constrained Environments

Framework: PyTorch

Dataset\*\*: CIFAR-10

Novelty: DDIM (up to 100× faster sampling)



## Overview

Project Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) and integrates Denoising Diffusion Implicit Models (DDIM) to accelerate image generation in resource-constrained environments.

Diffusion models have achieved state-of-the-art results in image synthesis but require a large number of denoising steps during inference, resulting in high computational costs. This project investigates how DDIM can significantly reduce inference time while maintaining acceptable image quality.

The implementation is developed using PyTorch and evaluated on the CIFAR-10 dataset.



This project:

1. Implements a **baseline DDPM** on CIFAR-10 (32×32 RGB) with a lightweight U-Net (\~28 M params)
2. Integrates **DDIM** (Song et al., ICLR 2021) as the algorithmic novelty — reduces sampling to as few as **10 steps** without retraining
3. Rigorously evaluates the **speed-up vs. quality trade-off** using FID + Inception Score
4. Runs a **fidelity–diversity ablation** across T = 1 000 → 100 → 10 steps



## Project Objectives

The project addresses the following objectives:

1.Implement a baseline DDPM model.

2.Integrate DDIM as an accelerated sampling strategy.

3.Measure computational efficiency and generation quality.

4.Analyze the trade-off between sampling speed and image fidelity.

5.Conduct an ablation study with different numbers of sampling steps.



## Dataset

CIFAR-10

•60,000 RGB images

•Resolution: 32 × 32

•10 object classes

•Training samples: 50,000

•Test samples: 10,000



Example classes include:

•Airplane

•Automobile

•Bird

•Cat

•Deer

•Dog

•Frog

•Horse

•Ship

•Truck



## Architecture

### U-Net (noise predictor ε\_θ)


\[B, 3, 32, 32]
    ↓  init\_conv 7×7
    ↓  Encoder ×4:  ResBlock → ResBlock → Downsample
       channels:    64 → 128 → 256 → 512
    ↓  Bottleneck:  ResBlock → SelfAttention → ResBlock
    ↓  Decoder ×4:  \[x ⊕ skip] → ResBlock → ResBlock → Upsample
    ↓  out\_conv 1×1
\[B, 3, 32, 32]  (predicted noise ε̂)


* **Timestep conditioning** — sinusoidal PE → MLP → AdaGN (scale + shift per ResBlock)
* **Self-attention** at bottleneck captures global structure
* **EMA shadow model** used exclusively at inference

### 

Training objective
L = E\_{t, x₀, ε} \[ ‖ε − ε\_θ(xₜ, t)‖² ]    (L\_simple, Ho et al. 2020)




## DDIM — Algorithmic Novelty

DDIM reformulates the reverse process as a **non-Markovian ODE**:


x\_{t-1} = √ᾱ\_{t-1} · pred\_x₀
         + √(1 − ᾱ\_{t-1} − σ²) · ε\_θ(xₜ, t)
         + σ · ε

σ = η · √\[(1−ᾱ\_{t-1})/(1−ᾱₜ)] · √\[1 − ᾱₜ/ᾱ\_{t-1}]


|η|Behaviour|
|-|-|
|0|Fully deterministic — same noise → same image every time|
|1|Recovers DDPM stochasticity|

**Key advantage**: any sub-sequence of S ≪ T timesteps can be used at inference with the *same trained weights*. No retraining required.

## 

## Running the Project

Open the notebook:

jupyter notebook project5\_lightweight\_ddpm.ipynb

Run all cells sequentially:

1.Imports

2.Globals

3.Utils

4.Data

5.Network

6.Train

7.Test



## Expected Results

|Method|Steps|Time (s)\*|Speed-up|FID ↓|IS ↑|
|-|-|-|-|-|-|
|DDPM (baseline)|1 000|\~120|1×|\~45|\~7.5|
|DDIM η=0|100|\~12|\~10×|\~48|\~7.3|
|DDIM η=0|10|\~1.2|\~100×|\~65|\~6.8|



## References

1. Ho, J., Jain, A., \& Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
2. Song, J., Meng, C., \& Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021.
3. Nichol, A. Q., \& Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
4. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022.
5. Salimans, T., \& Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR 2022.



## Author

•Yuriy Fedoryuk



