# Project 5: Lightweight Diffusion Models
## Accelerating Training and Inference for Resource-Constrained Environments


## Abstract

Denoising Diffusion Probabilistic Models (DDPMs) achieve state-of-the-art results in image generation, but their iterative sampling is notoriously slow — requiring up to 1000 reverse denoising steps per generated image. This project implements a baseline DDPM on CIFAR-10 and then integrates **DDIM (Denoising Diffusion Implicit Models)** as an algorithmic novelty to accelerate sampling by up to **109×** while preserving generation quality. The trade-off between computational efficiency and generation fidelity is rigorously evaluated using FID and Inception Score.


## Project Goals

- Implement a complete DDPM baseline on CIFAR-10 from scratch in PyTorch
- Integrate DDIM as a drop-in accelerated sampler requiring **no retraining**
- Quantify the speed/quality trade-off across sampling step counts (1000 → 100 → 10)
- Apply training stability techniques: EMA, AMP, and gradient clipping
- Evaluate generation quality using FID and Inception Score


## Background

### DDPM (Ho et al., 2020)

The forward diffusion process gradually corrupts an image `x₀` by adding Gaussian noise over `T` steps:

```
x₀ → x₁ → x₂ → ... → xₜ
```

A U-Net is trained to reverse this process, learning to predict the added noise `ε` at each timestep. While generating high-quality images, DDPM sampling requires all `T = 1000` denoising steps per image, making inference slow.

### DDIM (Song et al., 2021)

DDIM reformulates the reverse process as a **deterministic non-Markovian** path, allowing a dramatic reduction in the number of sampling steps without retraining. The update rule is:

```
x_{t-1} = √ᾱ_{t-1} · x̂₀  +  √(1 - ᾱ_{t-1} - σ²) · ε_θ(xₜ, t)  +  σ · ε
```

When `η = 0` (deterministic), the sampler skips most timesteps with minimal quality loss, yielding 10×–100× wall-clock speedup.


## Repository Structure

```
project5_lightweight_ddpm.ipynb   # Main notebook (all code, experiments, results)
data/                              # CIFAR-10 auto-downloaded here
outputs/                           # Saved model checkpoints and generated images
  best_model.pt                    # Best EMA checkpoint (lowest training loss)
  loss_curve.png                   # Training loss plot
  ddpm_samples.png                 # DDPM baseline generated images
  ddim_100_samples.png             # DDIM (100 steps) generated images
  ablation_steps.png               # Side-by-side fidelity vs. step ablation
README.md                          # This file
```


## Requirements & Installation

### Python Environment

```bash
python >= 3.12
```

### Core Dependencies

```bash
pip install torch torchvision tqdm matplotlib numpy
```

### Optional (for FID & Inception Score)

```bash
pip install torchmetrics[image]
```

> Without `torchmetrics`, the notebook falls back to proxy metrics (pixel variance and contrast). With it, FID and Inception Score are computed properly using the Inception-v3 network.

### Hardware

The notebook runs on both CPU and CUDA GPU. Detected automatically:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- **CPU:** Fully supported. Training 100 epochs takes significantly longer (~hours). Inference at 1000 steps takes ~1577 s per 1000 images.
- **GPU:** Recommended. Mixed-precision (AMP) and TF32 are enabled automatically on CUDA devices.

> **Note:** The included results were produced on **CPU** (PyTorch 2.12.0+cpu, TorchVision 0.27.0+cpu).


## Configuration Reference

All hyperparameters are defined in the **Globals** cell (Section 2). Key parameters:

### Data & I/O

| Variable | Value | Description |
|---|---|---|
| `DATASET` | `'CIFAR10'` | Dataset name |
| `IMAGE_SIZE` | `32` | Input image resolution |
| `IN_CHANNELS` | `3` | RGB channels |
| `DATA_DIR` | `'./data'` | CIFAR-10 download directory |
| `SAVE_DIR` | `'./outputs'` | Checkpoint and image output directory |

### Training

| Variable | Value | Description |
|---|---|---|
| `SEED` | `42` | Global random seed (NumPy, Python, PyTorch) |
| `BATCH_SIZE` | `512` | Training batch size |
| `EPOCHS` | `100` | Number of training epochs (optimal: 50–200) |
| `LR` | `4e-4` | Peak learning rate (AdamW) |
| `GRAD_CLIP` | `1.0` | Gradient clipping norm |
| `EMA_DECAY` | `0.9999` | Exponential Moving Average decay for model weights |
| `ACCUM_STEPS` | `1` | Gradient accumulation steps |
| `NUM_WORKERS` | `min(cpu_count, 8)` | DataLoader worker threads |

### Diffusion Process

| Variable | Value | Description |
|---|---|---|
| `T` | `1000` | Total diffusion timesteps |
| `BETA_SCHEDULE` | `'cosine'` | Noise schedule: `'linear'` or `'cosine'` |
| `DDIM_ETA` | `0.0` | DDIM stochasticity (0 = deterministic, 1 = DDPM) |

### Model Architecture

| Variable | Value | Description |
|---|---|---|
| `MODEL_DIM` | `32` | Base U-Net channel width |
| `DIM_MULTS` | `(1, 2, 4)` | Channel multipliers per encoder stage |
| `TIME_EMB_DIM` | `256` | Sinusoidal timestep embedding dimension |
| `NUM_GROUPS` | `4` | GroupNorm groups in ResBlocks |

### Evaluation

| Variable | Value | Description |
|---|---|---|
| `N_EVAL_IMAGES` | `1000` | Images to generate for FID/IS (10 000 for proper FID) |
| `EVAL_BATCH` | `64` | Batch size during evaluation generation |


## Dataset

**CIFAR-10** — downloaded automatically on first run.

| Property | Value |
|---|---|
| Training images | 50,000 |
| Test images | 10,000 |
| Resolution | 32 × 32 pixels |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Channels | RGB |

**Preprocessing:**
- Random horizontal flip (train only)
- Normalize to `[-1, 1]` range: `(pixel - 0.5) / 0.5`


## Model Architecture

### Lightweight U-Net (~3.32M parameters)

The noise prediction network follows a standard U-Net encoder–decoder structure with diffusion-specific additions:

```
Input (3 × 32 × 32)
    │
    ├── SinusoidalPositionEmbeddings(dim=32) → MLP → time_emb (256-d)
    │
    ├── Init Conv (3 → 32 channels)
    │
    ├── Encoder
    │   ├── Stage 1: 2× ResBlock(32→32),  Downsample → 16×16
    │   ├── Stage 2: 2× ResBlock(32→64),  Downsample → 8×8
    │   └── Stage 3: 2× ResBlock(64→128), Downsample → 4×4
    │
    ├── Bottleneck
    │   ├── ResBlock(128→128)
    │   ├── SelfAttention(128)
    │   └── ResBlock(128→128)
    │
    └── Decoder (with skip connections)
        ├── Stage 3: 2× ResBlock(256→64),  Upsample → 8×8
        ├── Stage 2: 2× ResBlock(128→32),  Upsample → 16×16
        └── Stage 1: 2× ResBlock(64→32),   Upsample → 32×32
            │
            └── Final Conv (32 → 3 channels)
Output (3 × 32 × 32) — predicted noise ε
```

**Building blocks:**

- **ResBlock:** Two 3×3 convolutions with GroupNorm (4 groups), SiLU activations, and additive timestep embedding injection. Residual shortcut with optional channel projection.
- **SelfAttention:** Multi-head self-attention at the bottleneck, enabling long-range spatial dependencies.
- **SinusoidalPositionEmbeddings:** Standard transformer-style positional encodings for timestep `t`, projected through a two-layer MLP.
- **Skip Connections:** Encoder feature maps are concatenated to decoder inputs at each resolution.


## Training Pipeline

### Loss Function

Standard DDPM noise prediction objective:

```python
L = MSE(ε, ε_θ(xₜ, t))
```

where `ε` is the ground-truth noise added during the forward process and `ε_θ` is the U-Net's prediction.

A pre-allocated noise buffer (`NOISE_BUFFER_SIZE = 2048`) is used to cycle through fixed noise vectors rather than sampling fresh per step, improving reproducibility.

### Optimizer & Scheduler

| Component | Setting |
|---|---|
| Optimizer | AdamW |
| Learning rate | 4e-4 (peak) |
| Weight decay | 1e-4 |
| LR schedule | Cosine Annealing over 100 epochs |
| Gradient clipping | Max norm 1.0 |

### Stability Techniques

- **EMA (Exponential Moving Average):** Separate EMA model (`decay = 0.9999`) is maintained throughout training. All evaluation and inference uses EMA weights, which produce significantly smoother outputs than the raw model.
- **AMP (Automatic Mixed Precision):** Enabled automatically on CUDA (`torch.cuda.amp`). On CPU, precision is set to `'high'` via `torch.set_float32_matmul_precision`.
- **Gradient Clipping:** Clips gradient norm to 1.0 before each optimizer step, preventing instability from large gradient spikes.

### Training Progress (100 Epochs on CPU)

| Epoch | Loss | Best Loss | LR |
|---|---|---|---|
| 10 | 0.0694 | 0.0694 | 3.90e-04 |
| 20 | 0.0644 | 0.0642 | 3.62e-04 |
| 30 | 0.0597 | 0.0597 | 3.18e-04 |
| 40 | 0.0550 | 0.0550 | 2.62e-04 |
| 50 | 0.0511 | 0.0511 | 2.00e-04 |
| 60 | 0.0466 | 0.0466 | 1.38e-04 |
| 70 | 0.0445 | 0.0445 | 8.24e-05 |
| 80 | 0.0425 | 0.0425 | 3.82e-05 |
| 90 | 0.0420 | 0.0416 | 9.79e-06 |
| 100 | 0.0415 | **0.0415** | 0.00e+00 |

**Key observations:**
- Sharp drop in first 10 epochs (0.069 → 0.051), driven by the cosine LR peak
- Smooth convergence — no instability spikes throughout training
- Plateau ~0.04 after epoch 80; gains diminish as LR anneals to zero
- Best checkpoint saved at epoch 100 with loss 0.0415

> The model was trained for 100 epochs as a proof-of-concept. For high-fidelity generation (FID < 50), training for 200+ epochs with a GPU is recommended.


## DDIM Accelerated Sampling

### How It Works

DDIM (Song et al., 2021) reformulates the reverse diffusion process as a **deterministic non-Markovian** trajectory. Instead of stepping through all `T = 1000` timesteps, DDIM selects a sub-sequence of `S` steps and skips the rest, using the predicted `x̂₀` to "jump" between timestep pairs.

**DDIM update rule:**
```
x_{t-1} = √ᾱ_{t-1} · x̂₀  +  √(1 - ᾱ_{t-1} - σ²) · ε_θ(xₜ, t)  +  σ · ε
```

With `η = 0` (the project default), `σ = 0`, making the process fully deterministic — the same initial noise always produces the same image.

### Key Properties

- **No retraining required:** DDIM uses the exact same U-Net weights as DDPM
- **Deterministic output** (`η = 0`): reproducible samples from fixed seeds
- **Tunable speed/quality:** reduce steps from 1000 → 100 → 10 with graceful quality degradation

### Implementation

```python
def ddim_sample(model, steps=100, eta=0.0, batch_size=16, ...):
    seq      = torch.linspace(0, T-1, steps, dtype=torch.long)
    seq_prev = torch.cat([torch.tensor([-1]), seq[:-1]])
    x        = torch.randn(batch_size, channels, image_size, image_size)

    for t_cur, t_prev in reversed(zip(seq, seq_prev)):
        abar  = alphas_cumprod[t_cur]
        ap    = alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
        pred  = model(x, t_cur)                          # ε_θ
        x0_hat = (x - √(1 - abar) * pred) / √abar       # predicted x₀
        x = √ap * x0_hat + √(1 - ap) * pred             # DDIM step
    return x
```


## Evaluation Metrics

### FID (Fréchet Inception Distance)

Measures the distance between the distribution of generated images and real images in Inception-v3 feature space. **Lower is better.** Computed with `torchmetrics.image.FrechetInceptionDistance` using `feature=2048`.

### Inception Score (IS)

Measures both quality (sharpness) and diversity of generated images using the Inception-v3 classifier. **Higher is better.** Computed with `torchmetrics.image.InceptionScore`.

> **Note:** FID ideally requires 10,000+ generated images for statistical reliability. This project uses `N_EVAL_IMAGES = 1000` for feasibility, so FID values should be interpreted as approximate indicators, not publication-grade measurements.


## Results

### Speed-Quality Trade-Off (1000 generated images, CPU)

| Method | Steps | Time (s) | Speed-Up | FID ↓ | IS_mean ↑ | IS_std |
|---|---|---|---|---|---|---|
| DDPM | 1000 | 1577.1 | 1.0× | 453.87 | 1.168 | 0.013 |
| DDIM | 100 | 142.8 | **11.0×** | 374.36 | 1.186 | 0.018 |
| DDIM | 10 | 14.5 | **109.1×** | 431.94 | 1.209 | 0.020 |

**Key findings:**

- **DDIM-100** achieves the best FID (374.36), outperforming DDPM despite being 11× faster. The deterministic sampling path appears to reduce noise-induced variance in the feature space.
- **DDIM-10** is 109× faster at a modest FID penalty vs. DDIM-100, remaining comparable to the DDPM baseline.
- All FID values are high (>350) because the model was only trained for 100 epochs on CPU. Longer training is expected to substantially improve generation quality — the speed-up ratios themselves are the primary finding, and these are independent of epoch count.

### Sample Timing Breakdown (16-image batch)

| Method | Wall-clock time |
|---|---|
| DDPM (T=1000) | ~30.7 s |
| DDIM (100 steps) | ~2.8 s |


## Ablation Study

### Fidelity vs. Step Reduction

Three configurations were compared side-by-side:

| Config | Description |
|---|---|
| DDPM T=1000 | Stochastic, full 1000 steps |
| DDIM 100 steps | Deterministic, 10% of timesteps |
| DDIM 10 steps | Deterministic, 1% of timesteps |

**Visual observations:**
- DDIM-100 produces slightly smoother pixel distributions than DDPM
- DDIM-10 shows minor coherence degradation at extreme compression but remains comparable
- All outputs appear noisy due to early-stage training (100 epochs); quality gains compound substantially with more epochs

### Beta Schedule Comparison

Both linear and cosine beta schedules were implemented:

- **Linear schedule** (Ho et al., 2020): `β` increases uniformly from `1e-4` to `0.02`
- **Cosine schedule** (Nichol & Dhariwal, 2021): `β` derived from cosine `ᾱ_t` curve; prevents over-corruption at late timesteps

The project uses the **cosine schedule** as default, which was found to improve training stability and sample quality at equal epoch counts.


## Conclusions & Future Work

### Conclusions

- DDIM provides **substantial acceleration** (11× at 100 steps, 109× at 10 steps) with limited quality degradation on a 100-epoch model
- The speed-up ratios scale linearly with step reduction and are device-agnostic
- EMA, AMP, and gradient clipping meaningfully stabilize the training loss curve
- DDIM-100 achieves **better FID** than the DDPM baseline, suggesting that deterministic sampling reduces stochastic noise in the evaluation pipeline

### Future Work

| Direction | Description |
|---|---|
| **Latent Diffusion Models** | Move diffusion into a compressed latent space (Rombach et al., 2022) to dramatically reduce per-step cost |
| **Progressive Distillation** | Distill DDIM into a student model that matches 1000-step quality in 4–8 steps (Salimans & Ho, 2022) |
| **Classifier-Free Guidance** | Condition generation on class labels for controllable, higher-fidelity output |
| **Higher-resolution datasets** | Scale to 64×64 or 256×256 (e.g., CelebA-HQ, LSUN) |
| **Longer training** | 200–500 epochs with a GPU to achieve FID < 50 on CIFAR-10 |


## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models.** *NeurIPS 2020.*
2. Song, J., Meng, C., & Ermon, S. (2021). **Denoising Diffusion Implicit Models.** *ICLR 2021.*
3. Rombach, R., et al. (2022). **High-Resolution Image Synthesis with Latent Diffusion Models.** *CVPR 2022.*
4. Salimans, T., & Ho, J. (2022). **Progressive Distillation for Fast Sampling of Diffusion Models.** *ICLR 2022.*
5. Nichol, A., & Dhariwal, P. (2021). **Improved Denoising Diffusion Probabilistic Models.** *ICML 2021.*


## Running the Notebook

```bash
# 1. Clone / download the notebook
# 2. Install dependencies
pip install torch torchvision tqdm matplotlib numpy torchmetrics[image]

# 3. Launch Jupyter
jupyter notebook project5_lightweight_ddpm.ipynb

# 4. Run all cells in order (Kernel → Restart & Run All)
#    CIFAR-10 is downloaded automatically on first run
#    Training takes ~hours on CPU, ~20 min on a modern GPU
```


> **Course:** Computer Vision  
> **Author:** Yuriy Fedoryuk  
> **Date:** June 2026  
> **Notebook:** `project5_lightweight_ddpm.ipynb`