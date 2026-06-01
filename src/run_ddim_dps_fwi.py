#!/usr/bin/env python3
"""
run_ddim_dps_fwi.py — DDIM + DPS (Tweedie) baseline for FWI
=============================================================
这是标准的 diffusion inverse 方法：
  训练：无条件 DDPM，只学速度场的生成分布，不看 seismic 观测
  推理：DDIM reverse + DPS (Diffusion Posterior Sampling) 梯度引导

核心推理步骤（每个 DDIM step）：
  1. Tweedie 估计：  x̂₀ = (xₜ − √(1−ᾱₜ)·εθ(xₜ,t)) / √ᾱₜ
  2. FWI 梯度：      g  = ∇_xₜ ‖FWM(x̂₀) − d‖²  （链式法则到 xₜ）
  3. DPS 更新：      xₜ₋₁ = DDIM(xₜ, εθ) − ζ·g

对比对象：
  - run_alsd_fwi.py 的 A-LSD 方法：inverse 在训练时进入 distillation target
  - 本方法：inverse 只在推理时通过梯度引导（test-time correction）

Reference:
  Chung et al., "Diffusion Posterior Sampling" (2022) arXiv:2209.14687
  Applied to FWI: unconditional prior + FWI likelihood gradient.
"""

import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────── helpers (same as run_alsd_fwi.py) ──────────────────

def seed_all(seed):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def log_transform(x):
    return torch.log1p(torch.abs(x)) * torch.sign(x)


def seismic_norm(seis):
    lo = log_transform(torch.tensor(-30., device=seis.device, dtype=seis.dtype))
    hi = log_transform(torch.tensor( 60., device=seis.device, dtype=seis.dtype))
    return (log_transform(seis) - lo) / (hi - lo + 1e-6) * 2. - 1.


def to_raw_velocity(x_norm):
    return (x_norm.clamp(-1.2, 1.2) + 1.) * 1500. + 1500.


def pad_velocity(v, nbc):
    return F.pad(v, (nbc, nbc, nbc, nbc), mode="replicate")


def fwm_call(fwm, x, args):
    v = pad_velocity(to_raw_velocity(x), args.nbc)
    return fwm(v, args.nbc, args.dx, args.nt, args.dt, args.freq,
               args.sx, args.sz, args.gx, args.gz, args.sampling_rate)


def load_cached_dataset(cache_dir, n):
    """Load pre-generated (vel, seis) from gen_fwi_obs.py output."""
    vel_files  = sorted(glob.glob(os.path.join(cache_dir, "vel_*.pt")))
    seis_files = sorted(glob.glob(os.path.join(cache_dir, "seis_*.pt")))
    vels, seises = [], []
    for vf, sf in zip(vel_files, seis_files):
        vels.append(torch.load(vf))
        seises.append(torch.load(sf))
        if sum(v.shape[0] for v in vels) >= n:
            break
    vel  = torch.cat(vels,  0)[:n]
    seis = torch.cat(seises, 0)[:n]
    return vel, seis


def compact_seis(y):
    return y[:, :, ::10, ::7].flatten(1)


def load_velocities(root, n):
    files = sorted(glob.glob(os.path.join(root, "model*.npy")))
    xs = []
    for path in files:
        arr = np.load(path).astype("float32")
        xs.append(arr if arr.ndim == 4 else arr[:, None])
        if sum(a.shape[0] for a in xs) >= n:
            break
    x = np.concatenate(xs, 0)[:n]
    x = torch.from_numpy(x)
    if x.shape[-2:] != (70, 70):
        x = F.interpolate(x, (70, 70), mode="bilinear", align_corners=False)
    return ((x.clamp(1500, 4500) - 1500.) / 3000. * 2. - 1.)


@torch.no_grad()
def cache_obs_online(x_all, path, fwm, args, device):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    ys = []
    for i in range(0, x_all.shape[0], args.fwm_batch):
        xb = x_all[i:i + args.fwm_batch].to(device)
        ys.append(seismic_norm(fwm_call(fwm, xb, args)).cpu())
    y = torch.cat(ys, 0)
    torch.save(y, path)
    return y


# ──────────────────────────────── DDPM noise schedule ───────────────────────

def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas   = torch.linspace(beta_start, beta_end, T)
    alphas  = 1. - betas
    abar    = torch.cumprod(alphas, 0)
    return betas, alphas, abar          # all shape [T]


# ──────────────────────────────── U-Net score model ─────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, ch * 2))

    def forward(self, x, t_emb):
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)
        scale = scale.view(-1, scale.shape[-1], 1, 1)
        shift = shift.view(-1, shift.shape[-1], 1, 1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class UNetScore(nn.Module):
    """
    Unconditional ε-prediction U-Net.
    Input:  x_t [B,1,H,W],  t [B] (integer timestep)
    Output: ε_pred [B,1,H,W]
    """
    def __init__(self, ch=64, t_dim=128, depth=3):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.downs      = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.up_convs   = nn.ModuleList()
        self.up_projs   = nn.ModuleList()   # 1x1 to merge skip
        self.ups        = nn.ModuleList()
        ch_cur = ch
        skip_chs = []
        for _ in range(depth):
            self.downs.append(ResBlock(ch_cur, t_dim))
            skip_chs.append(ch_cur)
            self.down_convs.append(nn.Conv2d(ch_cur, ch_cur * 2, 3, stride=2, padding=1))
            ch_cur *= 2
        self.mid1 = ResBlock(ch_cur, t_dim)
        self.mid2 = ResBlock(ch_cur, t_dim)
        for sk in reversed(skip_chs):
            # upsample conv: ch_cur -> ch_cur//2  (no stride; we use F.interpolate)
            self.up_convs.append(nn.Conv2d(ch_cur, ch_cur // 2, 1))
            ch_cur //= 2
            # project skip concat ch_cur+sk -> ch_cur
            self.up_projs.append(nn.Conv2d(ch_cur + sk, ch_cur, 1))
            self.ups.append(ResBlock(ch_cur, t_dim))
        self.out = nn.Sequential(nn.GroupNorm(8, ch_cur), nn.SiLU(),
                                 nn.Conv2d(ch_cur, 1, 3, padding=1))

    def _t_emb(self, t):
        half = self.t_dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb  = torch.cat([emb.sin(), emb.cos()], -1)
        return self.t_mlp(emb)

    def forward(self, x, t):
        t_emb  = self._t_emb(t)
        h      = self.inp(x)
        skips  = []
        for down, dc in zip(self.downs, self.down_convs):
            h = down(h, t_emb)
            skips.append(h)
            h = dc(h)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for uc, proj, up, sk in zip(self.up_convs, self.up_projs, self.ups, reversed(skips)):
            h = uc(h)
            # upsample to exactly match skip spatial size
            h = F.interpolate(h, size=sk.shape[2:], mode="bilinear", align_corners=False)
            h = proj(torch.cat([h, sk], 1))
            h = up(h, t_emb)
        return self.out(h)


# ──────────────────────────────── training ──────────────────────────────────

def train_ddpm(model, train_x, args, device, betas, abar):
    """Standard DDPM training: predict noise ε."""
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.05)
    loader = DataLoader(TensorDataset(train_x), args.batch_size, shuffle=True, drop_last=True)
    T = len(betas)

    it = iter(loader)
    log_path = os.path.join(args.run_dir, "train_metrics.jsonl")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:    (x0,) = next(it)
        except: it = iter(loader); (x0,) = next(it)
        x0 = x0.to(device)
        b  = x0.shape[0]

        # sample diffusion timestep
        ts  = torch.randint(0, T, (b,), device=device)
        eps = torch.randn_like(x0)
        ab  = abar[ts].view(b, 1, 1, 1)
        xt  = ab.sqrt() * x0 + (1 - ab).sqrt() * eps

        eps_pred = model(xt, ts)
        loss     = F.mse_loss(eps_pred, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()

        if step % 200 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss)}
            print(json.dumps(row), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    print(f"DDPM training done in {time.time()-t0:.1f}s", flush=True)


# ──────────────────────────────── DPS inference ─────────────────────────────

def ddim_dps_sample(model, y_seis, fwm, args, device, betas, alphas, abar,
                    n_steps=50):
    """
    DDIM reverse with DPS (Diffusion Posterior Sampling) guidance.

    At each step:
      1. ε = εθ(xₜ, t)
      2. Tweedie:  x̂₀ = (xₜ − √(1−ᾱₜ)·ε) / √ᾱₜ
      3. FWI grad: g = ∇_xₜ ‖FWM(x̂₀) − d‖²  (through x̂₀)
      4. DDIM step: xₜ₋₁ = √ᾱₜ₋₁·x̂₀ + √(1−ᾱₜ₋₁)·ε_dir
      5. DPS:       xₜ₋₁ = xₜ₋₁ − ζ·g
    """
    T   = len(betas)
    b   = y_seis.shape[0]
    # pick n_steps evenly spaced timesteps
    ts  = list(reversed(range(0, T, T // n_steps)))[:n_steps]

    xt  = torch.randn(b, 1, 70, 70, device=device)

    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else 0
        tv     = torch.full((b,), t,      device=device, dtype=torch.long)
        tv_p   = torch.full((b,), t_prev, device=device, dtype=torch.long)

        ab_t  = abar[t].view(1, 1, 1, 1)
        ab_p  = abar[t_prev].view(1, 1, 1, 1) if t_prev > 0 else torch.zeros(1,1,1,1,device=device)

        # 1. score
        with torch.no_grad():
            eps = model(xt, tv)

        # 2. Tweedie x̂₀
        x0_hat = (xt - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
        x0_hat = x0_hat.clamp(-1.5, 1.5)

        # 3. FWI gradient  d/dxₜ ‖FWM(x̂₀(xₜ)) − d‖²
        # torch.enable_grad() needed: caller may be inside @torch.no_grad()
        with torch.enable_grad():
            xt_req = xt.detach().requires_grad_(True)
            ab_t_g = abar[t].view(1, 1, 1, 1)
            eps_g  = model(xt_req, tv)
            x0_g   = (xt_req - (1 - ab_t_g).sqrt() * eps_g) / ab_t_g.sqrt().clamp_min(1e-8)
            seis_p = fwm_call(fwm, x0_g.clamp(-1.5, 1.5), args)
            misfit = F.mse_loss(seismic_norm(seis_p), y_seis)
            (grad_xt,) = torch.autograd.grad(misfit, xt_req)

        # 4. DDIM deterministic step
        eps_dir = eps.detach()
        xt_new  = ab_p.sqrt() * x0_hat.detach() + (1 - ab_p).sqrt() * eps_dir

        # 5. DPS guidance
        g_norm = grad_xt.flatten(1).norm(dim=1).view(b,1,1,1).clamp_min(1e-8)
        xt     = (xt_new - args.dps_zeta * grad_xt.detach() / g_norm).clamp(-1.5, 1.5)

    return xt.clamp(-1, 1)


# ──────────────────────────────── evaluation ────────────────────────────────

@torch.no_grad()
def evaluate_ddpm(model, val_x, val_seis, fwm, args, device, abar, n_eval=8):
    model.eval()
    T   = len(abar)
    mse_raw, mse_guided = [], []
    for i in range(0, min(n_eval, val_x.shape[0]), args.batch_size):
        x0     = val_x[i:i + args.batch_size].to(device)
        y_seis = val_seis[i:i + args.batch_size].to(device)
        b      = x0.shape[0]

        # unconditional sample (no guidance)
        eps  = torch.randn_like(x0)
        ts   = torch.full((b,), T - 1, device=device, dtype=torch.long)
        ab   = abar[-1].view(1,1,1,1)
        xt   = ab.sqrt() * x0 + (1-ab).sqrt() * eps
        eps_pred = model(xt, ts)
        x0_unc   = ((xt - (1-ab).sqrt()*eps_pred)/ab.sqrt()).clamp(-1,1)
        mse_raw.append(F.mse_loss(x0_unc, x0).item())

        # DPS-guided sample
        x0_g = ddim_dps_sample(model, y_seis, fwm, args, device,
                                torch.zeros(T), torch.ones(T), abar, n_steps=20)
        mse_guided.append(F.mse_loss(x0_g, x0).item())

    model.train()
    return {"mse_unguided": float(np.mean(mse_raw)),
            "mse_dps_guided": float(np.mean(mse_guided))}


# ──────────────────────────────── main ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",   default="/data10/fwi_cache/fvb",
                   help="dir with vel_*.pt / seis_*.pt from gen_fwi_obs.py")
    p.add_argument("--vel_root",    default="/workspace/fdo-fwi/data/fvb/velocity",
                   help="fallback: generate obs online if cache_dir empty")
    p.add_argument("--out_dir",     default="/data10/alsd_fwi")
    p.add_argument("--n_train",     type=int, default=2000)
    p.add_argument("--n_val",       type=int, default=400)
    p.add_argument("--steps",       type=int, default=5000)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--fwm_batch",   type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--dps_n_steps", type=int,   default=50)
    p.add_argument("--dps_zeta",    type=float, default=0.3)
    # FWM
    p.add_argument("--nt",          type=int, default=300)
    p.add_argument("--sampling_rate", type=int, default=2)
    args = p.parse_args()

    args.nbc = 120; args.dx = 10; args.dt = 1e-3; args.freq = 15.
    args.sz  = 10;  args.gz = 10
    grids = 70
    args.sx = np.linspace(0, grids-1, 5)    * args.dx
    args.gx = np.linspace(0, grids-1, grids) * args.dx

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, "/workspace/fdo-fwi/data/fvb")
    from data_gen_f import FWM as fwm

    args.run_dir = os.path.join(args.out_dir, "ddim_dps")
    os.makedirs(args.run_dir, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    n_total = args.n_train + args.n_val
    cache_vel  = sorted(glob.glob(os.path.join(args.cache_dir, "vel_*.pt")))
    if cache_vel:
        print(f"Loading from cache: {len(cache_vel)} files", flush=True)
        vel, seis = load_cached_dataset(args.cache_dir, n_total)
    else:
        print("Cache not found, generating online (slow)…", flush=True)
        vel   = load_velocities(args.vel_root, n_total)
        cache = os.path.join(args.out_dir,
                    f"obs_nt{args.nt}_sr{args.sampling_rate}_n{n_total}.pt")
        seis  = cache_obs_online(vel, cache, fwm, args, device)

    print(f"Dataset: vel={vel.shape}  seis={seis.shape}", flush=True)

    perm    = torch.randperm(vel.shape[0])
    train_x = vel[perm[:args.n_train]]
    val_x   = vel[perm[args.n_train:]]
    val_s   = seis[perm[args.n_train:]]

    # ── DDPM schedule ─────────────────────────────────────────────────────────
    T = 1000
    betas, alphas, abar = make_schedule(T)
    betas   = betas.to(device)
    alphas  = alphas.to(device)
    abar    = abar.to(device)

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNetScore(ch=64, t_dim=128, depth=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNetScore params: {n_params/1e6:.2f}M", flush=True)

    # ── train ─────────────────────────────────────────────────────────────────
    train_ddpm(model, train_x, args, device, betas, abar)

    # ── evaluate DPS ──────────────────────────────────────────────────────────
    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.run_dir, "final.pt"))

    # evaluate on a small subset (DPS is slow)
    metrics = evaluate_ddpm(model, val_x, val_s, fwm, args, device, abar,
                            n_eval=min(16, args.n_val))
    metrics.update({"n_train": args.n_train, "steps": args.steps,
                    "dps_zeta": args.dps_zeta, "dps_n_steps": args.dps_n_steps})
    with open(os.path.join(args.run_dir, "final.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("FINAL", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
