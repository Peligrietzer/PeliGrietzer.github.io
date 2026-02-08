#!/usr/bin/env python3
"""
Diffusion-Model Analogue of Autoencoder Reconstruction Sampling
===============================================================

Tests whether "manifold-projected" samples from a trained DDPM transfer
sample-efficiency to student models, analogous to the known result for
autoencoders: f(f(D_m)_n) ≈ f(D_m) while f(D_n) is much worse.

The diffusion analogue of "reconstruction" is:
    clean image → noise to timestep t_proj → denoise back to t=0

This script runs the full experiment on Fashion-MNIST:
  1. Train a teacher DDPM on full training set (60k images)
  2. Generate projected (partial noise + denoise) and purely generated datasets
  3. Train student DDPMs on raw / projected / generated subsets
  4. Evaluate all models via FID

Usage:
    python ddpm_experiment.py                    # run full experiment
    python ddpm_experiment.py --quick            # quick smoke test
    python ddpm_experiment.py --teacher-only     # train teacher only
    python ddpm_experiment.py --skip-teacher     # load saved teacher, run students
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    # Paths
    output_dir="./results",
    data_dir="./data",

    # Noise schedule
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    schedule="linear",          # "linear" or "cosine"

    # U-Net architecture
    base_channels=64,
    channel_mults=(1, 2, 2),    # down/up block multipliers
    num_res_blocks=2,           # residual blocks per resolution
    attention_resolutions=(2,), # apply attention at these downsample levels (0-indexed)
    dropout=0.1,
    time_embed_dim=256,

    # Training
    lr=2e-4,
    batch_size=128,
    ema_decay=0.9999,
    grad_clip=1.0,

    # Teacher training
    teacher_epochs=80,
    teacher_patience=10,        # early stopping patience (epochs)

    # Student training
    student_steps=15000,        # fixed number of gradient steps per student

    # Projection / generation
    t_proj_values=(200,),       # timesteps for projection; tuple for sweep
    n_values=(500, 1000, 2000, 5000, 10000, 20000),

    # Evaluation
    eval_n_samples=10000,       # samples to generate for FID
    fid_batch_size=256,

    # DDPM sampling
    sampling_batch_size=256,

    # Misc
    seed=42,
    num_workers=4,
    log_every=200,              # log loss every N steps
)

QUICK_CONFIG_OVERRIDES = dict(
    teacher_epochs=3,
    teacher_patience=100,
    student_steps=300,
    n_values=(500, 1000),
    t_proj_values=(200,),
    eval_n_samples=1000,
    base_channels=32,
    num_res_blocks=1,
    attention_resolutions=(),
)


def get_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.quick:
        cfg.update(QUICK_CONFIG_OVERRIDES)
    # CLI overrides
    for k in ["output_dir", "data_dir", "seed", "teacher_epochs", "student_steps"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = type(cfg[k])(v) if k in cfg else v
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """Precomputes all diffusion coefficients for a linear or cosine schedule."""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 schedule="linear", device="cpu"):
        self.T = num_timesteps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        elif schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * (math.pi / 2)) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_cumprod[:-1]])

        # Store everything as float32 on device
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alpha_cumprod = alpha_cumprod.float().to(device)
        self.alpha_cumprod_prev = alpha_cumprod_prev.float().to(device)
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).float().to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).float().to(device)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        ).float().to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)
        ).float().to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: sample x_t given x_0."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """Recover x_0 prediction from predicted noise."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha

    def q_posterior_mean_variance(self, x0, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        mean = (self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x0 +
                self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t)
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        return mean, var, log_var


# ---------------------------------------------------------------------------
# U-Net Model
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResBlock(nn.Module):
    """Residual block with time embedding and optional dropout."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention at a given resolution."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each: B, heads, C//heads, HW
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for DDPM on 28×28 grayscale images.

    Architecture:
        - Sinusoidal time embedding → MLP
        - Encoder: repeated (ResBlock × num_res_blocks [+ Attention] + Downsample)
        - Middle: ResBlock + Attention + ResBlock
        - Decoder: repeated (ResBlock × num_res_blocks [+ Attention] + Upsample)
        - Skip connections between encoder and decoder at each resolution
    """

    def __init__(self, in_ch=1, base_ch=64, ch_mults=(1, 2, 2),
                 num_res_blocks=2, attn_resolutions=(2,), dropout=0.1,
                 time_embed_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        channels = [base_ch]
        ch = base_ch
        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_embed_dim, dropout)]
                if level in attn_resolutions:
                    layers.append(AttentionBlock(out_ch))
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(out_ch)
                ch = out_ch
            if level < len(ch_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                channels.append(ch)

        # --- Middle ---
        self.mid_block1 = ResBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_embed_dim, dropout)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_embed_dim, dropout)]
                if level in attn_resolutions:
                    layers.append(AttentionBlock(out_ch))
                self.up_blocks.append(nn.ModuleList(layers))
                ch = out_ch
            if level > 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))

        # Final
        self.final_norm = nn.GroupNorm(min(32, ch), ch)
        self.final_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)
        skips = [h]

        # Encoder
        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
                skips.append(h)
            else:
                for layer in block:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for block in self.up_blocks:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                s = skips.pop()
                h = torch.cat([h, s], dim=1)
                for layer in block:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)

        h = self.final_conv(F.silu(self.final_norm(h)))
        return h


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) for model weights
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model):
        """Replace model params with EMA params; returns backup."""
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup)

    def state_dict(self):
        return dict(self.shadow)

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_fashion_mnist(data_dir, train=True):
    """Returns Fashion-MNIST as float tensors in [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] → [-1,1]
    ])
    ds = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    return ds


def dataset_to_tensors(ds):
    """Extract all images (and labels) from a dataset as tensors."""
    loader = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=0)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
    return torch.cat(imgs), torch.cat(labels)


def make_subset_loader(images, batch_size, shuffle=True):
    """DataLoader from a tensor of images."""
    ds = TensorDataset(images)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=True, num_workers=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ddpm(model, dataloader, noise_schedule, cfg, device,
               num_epochs=None, num_steps=None, desc="Training",
               val_loader=None, patience=None):
    """
    Train a DDPM model. Supports either epoch-based (with optional early
    stopping on validation loss) or step-based training.

    Returns: dict with training history.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    ema = EMA(model, decay=cfg["ema_decay"])
    ns = noise_schedule
    T = ns.T

    history = {"train_loss": [], "val_loss": [], "steps": []}
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    def train_one_step(batch):
        nonlocal global_step
        x0 = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t, _ = ns.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        if cfg["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        ema.update(model)
        global_step += 1
        return loss.item()

    if num_epochs is not None:
        # Epoch-based training (for teacher)
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"{desc} epoch {epoch+1}/{num_epochs}", leave=False)
            for batch in pbar:
                loss_val = train_one_step(batch)
                epoch_losses.append(loss_val)
                if global_step % cfg["log_every"] == 0:
                    history["train_loss"].append(loss_val)
                    history["steps"].append(global_step)
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_train = np.mean(epoch_losses)

            # Validation
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x0 = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        B = x0.shape[0]
                        t = torch.randint(0, T, (B,), device=device)
                        noise = torch.randn_like(x0)
                        x_t, _ = ns.q_sample(x0, t, noise)
                        pred_noise = model(x_t, t)
                        val_losses.append(F.mse_loss(pred_noise, noise).item())
                avg_val = np.mean(val_losses)
                history["val_loss"].append(avg_val)
                print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

                # Early stopping
                if patience is not None:
                    if avg_val < best_val_loss - 1e-5:
                        best_val_loss = avg_val
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break
            else:
                print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}")

    elif num_steps is not None:
        # Step-based training (for students)
        model.train()
        data_iter = iter(dataloader)
        pbar = tqdm(range(num_steps), desc=desc, leave=False)
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            loss_val = train_one_step(batch)
            if step % cfg["log_every"] == 0:
                history["train_loss"].append(loss_val)
                history["steps"].append(global_step)
            if step % 500 == 0:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

    return history, ema


# ---------------------------------------------------------------------------
# Sampling (reverse diffusion)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(model, noise_schedule, shape, device, batch_size=256):
    """Generate samples via full DDPM reverse process (t=T → t=0)."""
    ns = noise_schedule
    n_total = shape[0]
    all_samples = []
    for start in range(0, n_total, batch_size):
        B = min(batch_size, n_total - start)
        x = torch.randn(B, *shape[1:], device=device)
        for t in reversed(range(ns.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = model(x, t_batch)
            x0_pred = ns.predict_x0_from_eps(x, t_batch, eps_pred)
            x0_pred = x0_pred.clamp(-1, 1)
            mean, var, log_var = ns.q_posterior_mean_variance(x0_pred, x, t_batch)
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = mean + torch.exp(0.5 * log_var) * noise
        all_samples.append(x.cpu())
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def ddpm_project(model, noise_schedule, images, t_proj, device, batch_size=256):
    """
    'Reconstruct' images via partial noising + denoising.
    This is the diffusion analogue of autoencoder reconstruction:
        x_0 → noise to t_proj → denoise back to t=0
    """
    ns = noise_schedule
    all_out = []
    for start in range(0, len(images), batch_size):
        x0 = images[start:start + batch_size].to(device)
        B = x0.shape[0]

        # Forward: noise to t_proj
        t_batch = torch.full((B,), t_proj, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t, _ = ns.q_sample(x0, t_batch, noise)

        # Reverse: denoise from t_proj back to 0
        x = x_t
        for t in reversed(range(t_proj)):
            tb = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = model(x, tb)
            x0_pred = ns.predict_x0_from_eps(x, tb, eps_pred)
            x0_pred = x0_pred.clamp(-1, 1)
            mean, var, log_var = ns.q_posterior_mean_variance(x0_pred, x, tb)
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = mean + torch.exp(0.5 * log_var) * noise
        all_out.append(x.cpu())
    return torch.cat(all_out, dim=0)


# ---------------------------------------------------------------------------
# FID Computation
# ---------------------------------------------------------------------------

def save_images_for_fid(images, out_dir):
    """Save tensor images as PNGs for FID computation."""
    from torchvision.utils import save_image
    os.makedirs(out_dir, exist_ok=True)
    # images in [-1, 1] → [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    for i, img in enumerate(images):
        save_image(img, os.path.join(out_dir, f"{i:05d}.png"))


def compute_fid(generated_dir, reference_dir, device_str="cuda"):
    """Compute FID using clean-fid."""
    from cleanfid import fid
    score = fid.compute_fid(generated_dir, reference_dir,
                            mode="clean",
                            device=torch.device(device_str))
    return score


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def build_model(cfg, device):
    model = UNet(
        in_ch=1,
        base_ch=cfg["base_channels"],
        ch_mults=cfg["channel_mults"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_resolutions=cfg["attention_resolutions"],
        dropout=cfg["dropout"],
        time_embed_dim=cfg["time_embed_dim"],
    ).to(device)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(args):
    cfg = get_config(args)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, f, indent=2)

    # ---- Data ----
    print("\n=== Loading Fashion-MNIST ===")
    train_ds = get_fashion_mnist(cfg["data_dir"], train=True)
    test_ds = get_fashion_mnist(cfg["data_dir"], train=False)
    all_train_imgs, all_train_labels = dataset_to_tensors(train_ds)
    test_imgs, test_labels = dataset_to_tensors(test_ds)
    print(f"Training images: {all_train_imgs.shape}")
    print(f"Test images: {test_imgs.shape}")

    # Split: 54k train, 6k val (for teacher)
    n_val = 6000
    perm = torch.randperm(len(all_train_imgs))
    val_imgs = all_train_imgs[perm[:n_val]]
    teacher_train_imgs = all_train_imgs[perm[n_val:]]
    print(f"Teacher train: {teacher_train_imgs.shape}, Val: {val_imgs.shape}")

    # Noise schedule
    ns = NoiseSchedule(
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        schedule=cfg["schedule"],
        device=device,
    )

    # ---- Reference images for FID ----
    ref_dir = output_dir / "fid_reference"
    if not (ref_dir / "00000.png").exists():
        print("Saving reference images for FID...")
        save_images_for_fid(test_imgs, str(ref_dir))

    # ================================================================
    # Step 1: Teacher
    # ================================================================
    teacher_path = output_dir / "teacher.pt"

    if args.skip_teacher and teacher_path.exists():
        print("\n=== Loading saved teacher ===")
        teacher = build_model(cfg, device)
        ckpt = torch.load(teacher_path, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt["ema"])
        print(f"Teacher loaded from {teacher_path}")
    else:
        print("\n=== Step 1: Training Teacher DDPM ===")
        teacher = build_model(cfg, device)
        print(f"Teacher parameters: {count_params(teacher):,}")

        train_loader = make_subset_loader(teacher_train_imgs, cfg["batch_size"])
        val_loader = make_subset_loader(val_imgs, cfg["batch_size"], shuffle=False)

        t0 = time.time()
        history, ema = train_ddpm(
            teacher, train_loader, ns, cfg, device,
            num_epochs=cfg["teacher_epochs"],
            desc="Teacher",
            val_loader=val_loader,
            patience=cfg["teacher_patience"],
        )
        elapsed = time.time() - t0
        print(f"Teacher training time: {elapsed/60:.1f} min")

        # Apply EMA weights for all downstream use
        ema.apply(teacher)

        # Save
        torch.save({
            "model": teacher.state_dict(),
            "ema": teacher.state_dict(),  # already applied
            "history": history,
            "config": cfg,
        }, teacher_path)
        print(f"Teacher saved to {teacher_path}")

        # Plot training loss
        if history["train_loss"]:
            plt.figure(figsize=(8, 4))
            plt.plot(history["steps"], history["train_loss"], alpha=0.5, label="train")
            if history["val_loss"]:
                val_x = np.linspace(0, history["steps"][-1], len(history["val_loss"]))
                plt.plot(val_x, history["val_loss"], label="val")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Teacher Training Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "teacher_loss.png", dpi=150)
            plt.close()

    if args.teacher_only:
        print("--teacher-only mode: exiting after teacher training.")
        return

    # ---- Teacher FID ----
    print("\n=== Evaluating teacher FID ===")
    teacher.eval()
    teacher_samples_dir = output_dir / "samples_teacher"
    if not (teacher_samples_dir / "00000.png").exists():
        print(f"Generating {cfg['eval_n_samples']} teacher samples...")
        teacher_samples = ddpm_sample(
            teacher, ns, (cfg["eval_n_samples"], 1, 28, 28), device,
            batch_size=cfg["sampling_batch_size"],
        )
        save_images_for_fid(teacher_samples, str(teacher_samples_dir))
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_fid = compute_fid(str(teacher_samples_dir), str(ref_dir), device_str)
    print(f"Teacher FID: {teacher_fid:.2f}")

    # ================================================================
    # Step 2 & 3: Generate projected/generated data, train students
    # ================================================================

    results = {"teacher_fid": teacher_fid, "experiments": []}
    n_values = cfg["n_values"]
    t_proj_values = cfg["t_proj_values"]

    for t_proj in t_proj_values:
        print(f"\n{'='*60}")
        print(f"t_proj = {t_proj}")
        print(f"{'='*60}")

        for n in n_values:
            print(f"\n--- n = {n}, t_proj = {t_proj} ---")
            row = {"n": n, "t_proj": t_proj}

            # Subsample raw data
            raw_idx = torch.randperm(len(all_train_imgs))[:n]
            raw_imgs = all_train_imgs[raw_idx]

            # Generate projected data
            print(f"  Projecting {n} images at t_proj={t_proj}...")
            proj_imgs = ddpm_project(teacher, ns, raw_imgs, t_proj, device,
                                     batch_size=cfg["sampling_batch_size"])

            # Generate purely generated data
            print(f"  Generating {n} images from scratch...")
            gen_imgs = ddpm_sample(teacher, ns, (n, 1, 28, 28), device,
                                   batch_size=cfg["sampling_batch_size"])

            # Train three students
            for condition, data in [("raw", raw_imgs), ("proj", proj_imgs), ("gen", gen_imgs)]:
                print(f"  Training student_{condition}_{n}...")
                student = build_model(cfg, device)
                loader = make_subset_loader(data, min(cfg["batch_size"], n))

                _, ema = train_ddpm(
                    student, loader, ns, cfg, device,
                    num_steps=cfg["student_steps"],
                    desc=f"student_{condition}_{n}",
                )
                ema.apply(student)
                student.eval()

                # Generate samples for FID
                sample_dir = output_dir / f"samples_{condition}_{n}_t{t_proj}"
                print(f"  Generating {cfg['eval_n_samples']} samples for FID...")
                samples = ddpm_sample(
                    student, ns, (cfg["eval_n_samples"], 1, 28, 28), device,
                    batch_size=cfg["sampling_batch_size"],
                )
                save_images_for_fid(samples, str(sample_dir))
                fid_score = compute_fid(str(sample_dir), str(ref_dir), device_str)
                row[f"fid_{condition}"] = fid_score
                print(f"  FID {condition}: {fid_score:.2f}")

                # Cleanup to free GPU memory
                del student, samples
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            results["experiments"].append(row)

            # Intermediate save
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    # ================================================================
    # Step 4: Results table and plot
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Build table
    header = f"{'n':>7} | {'FID raw':>10} | {'FID proj':>10} | {'FID gen':>10} | {'Teacher':>10}"
    print(header)
    print("-" * len(header))
    for row in results["experiments"]:
        print(f"{row['n']:>7} | {row['fid_raw']:>10.2f} | {row['fid_proj']:>10.2f} | "
              f"{row['fid_gen']:>10.2f} | {teacher_fid:>10.2f}")

    # Save results table
    with open(output_dir / "results_table.txt", "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for row in results["experiments"]:
            f.write(f"{row['n']:>7} | {row['fid_raw']:>10.2f} | {row['fid_proj']:>10.2f} | "
                    f"{row['fid_gen']:>10.2f} | {teacher_fid:>10.2f}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ns_vals = [r["n"] for r in results["experiments"]]
    fid_raw = [r["fid_raw"] for r in results["experiments"]]
    fid_proj = [r["fid_proj"] for r in results["experiments"]]
    fid_gen = [r["fid_gen"] for r in results["experiments"]]

    ax.plot(ns_vals, fid_raw, "o-", label="Student (raw data)", color="#d62728", linewidth=2)
    ax.plot(ns_vals, fid_proj, "s-", label=f"Student (projected, t={t_proj_values[0]})",
            color="#2ca02c", linewidth=2)
    ax.plot(ns_vals, fid_gen, "^-", label="Student (generated)", color="#1f77b4", linewidth=2)
    ax.axhline(y=teacher_fid, linestyle="--", color="gray", alpha=0.7,
               label=f"Teacher (FID={teacher_fid:.1f})")

    ax.set_xlabel("Number of training samples (n)", fontsize=12)
    ax.set_ylabel("FID (lower is better)", fontsize=12)
    ax.set_title("Diffusion Manifold Projection: Sample Efficiency", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns_vals)
    ax.set_xticklabels([str(n) for n in ns_vals])
    plt.tight_layout()
    plt.savefig(output_dir / "fid_comparison.png", dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_dir / 'fid_comparison.png'}")

    # Save final results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nExperiment complete!")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDPM manifold projection experiment on Fashion-MNIST")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with tiny model and few steps")
    parser.add_argument("--teacher-only", action="store_true",
                        help="Only train the teacher model, then exit")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Load saved teacher and run student experiments only")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=None)
    parser.add_argument("--student-steps", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
