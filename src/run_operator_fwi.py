#!/usr/bin/env python3
"""
Neural-operator and physics-informed amortized FWI baselines.

Variants:
  unet        supervised observation-to-velocity map
  fno         supervised Fourier neural operator style map
  pinn_unet   supervised map plus occasional FWM data-consistency loss

All variants use the same cache format as the generative scripts.
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_t(x):
    return torch.log1p(torch.abs(x)) * torch.sign(x)


def seismic_norm(seis):
    lo = log_t(torch.tensor(-30.0, device=seis.device, dtype=seis.dtype))
    hi = log_t(torch.tensor(60.0, device=seis.device, dtype=seis.dtype))
    return (log_t(seis) - lo) / (hi - lo + 1e-6) * 2.0 - 1.0


def to_raw(x):
    return (x.clamp(-1.2, 1.2) + 1.0) * 1500.0 + 1500.0


def pad_v(v, nbc):
    return F.pad(v, (nbc, nbc, nbc, nbc), mode="replicate")


def fwm_call(fwm, x, args):
    return fwm(
        pad_v(to_raw(x), args.nbc),
        args.nbc,
        args.dx,
        args.nt,
        args.dt,
        args.freq,
        args.sx,
        args.sz,
        args.gx,
        args.gz,
        args.sampling_rate,
    )


def per_sample_misfit(pred_seis, obs_seis):
    return (seismic_norm(pred_seis) - obs_seis).flatten(1).pow(2).mean(dim=1)


def load_cache(cache_dir, n):
    vf = sorted(glob.glob(os.path.join(cache_dir, "vel_*.pt")))
    sf = sorted(glob.glob(os.path.join(cache_dir, "seis_*.pt")))
    if not vf or not sf:
        raise FileNotFoundError(f"No vel_*.pt/seis_*.pt in {cache_dir}")
    vs, ss = [], []
    for v, s in zip(vf, sf):
        vs.append(torch.load(v).detach().clone())
        ss.append(torch.load(s).detach().clone())
        if sum(x.shape[0] for x in vs) >= n:
            break
    return torch.cat(vs, 0)[:n], torch.cat(ss, 0)[:n]


def obs_image(seis):
    # Collapse source dimension into channels and resize time/receiver image to 70x70.
    b, src, nt, nr = seis.shape
    y = F.interpolate(seis, size=(70, 70), mode="bilinear", align_corners=False)
    return y.reshape(b, src, 70, 70)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=5, base=48):
        super().__init__()
        self.e1 = ConvBlock(cin, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, y):
        e1 = self.e1(y)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return torch.tanh(self.out(d1))


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes1=16, modes2=16):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (cin * cout)
        self.w = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, 2))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.w.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        weight = torch.view_as_complex(self.w)
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bcxy,coxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], weight
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, cin=5, width=48, depth=4):
        super().__init__()
        self.lift = nn.Conv2d(cin + 2, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, y):
        b, _, h, w = y.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=y.device),
            torch.linspace(0, 1, w, device=y.device),
            indexing="ij",
        )
        grid = torch.stack([yy, xx], dim=0).expand(b, -1, -1, -1)
        x = self.lift(torch.cat([y, grid], dim=1))
        for s, l in zip(self.spec, self.local):
            x = F.gelu(s(x) + l(x))
        return torch.tanh(self.proj(x))


@torch.no_grad()
def evaluate(model, loader, fwm, args, device, n_eval):
    model.eval()
    rows = []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(obs_image(y)).clamp(-1, 1)
        mse = F.mse_loss(pred, x, reduction="none").flatten(1).mean(dim=1)
        mis = per_sample_misfit(fwm_call(fwm, pred, args), y)
        for a, b in zip(mse.cpu().tolist(), mis.cpu().tolist()):
            rows.append({"mse": a, "misfit": b})
        seen += x.shape[0]
        if seen >= n_eval:
            break
    out = {}
    for k in rows[0].keys():
        vals = np.array([r[k] for r in rows[:n_eval]], dtype=np.float64)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["unet", "fno", "pinn_unet"], default="unet")
    p.add_argument("--dataset", choices=["fvb", "cva", "cfa", "sta"], default="fvb")
    p.add_argument("--cache_dir", default="/data10/fwi_cache/fvb")
    p.add_argument("--out_dir", default="/workspace/fmm_outputs/operator_fwi")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val", type=int, default=400)
    p.add_argument("--n_eval", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--physics_weight", type=float, default=0.05)
    p.add_argument("--physics_every", type=int, default=10)
    p.add_argument("--nt", type=int, default=300)
    p.add_argument("--sampling_rate", type=int, default=2)
    args = p.parse_args()

    args.nbc = 120
    args.dx = 10
    args.dt = 1e-3
    args.freq = 15.0
    args.sz = 10
    args.gz = 10
    grids = 70
    args.sx = np.linspace(0, grids - 1, 5) * args.dx
    args.gx = np.linspace(0, grids - 1, grids) * args.dx

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, args.variant)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, os.path.join("/workspace/fdo-fwi/data", args.dataset))
    from data_gen_f import FWM as fwm

    vel, seis = load_cache(args.cache_dir, args.n_train + args.n_val)
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(vel.shape[0], generator=gen)
    tr = TensorDataset(vel[perm[: args.n_train]], seis[perm[: args.n_train]])
    va = TensorDataset(vel[perm[args.n_train : args.n_train + args.n_val]], seis[perm[args.n_train : args.n_train + args.n_val]])
    tr_loader = DataLoader(tr, args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va, args.batch_size, shuffle=False)

    model = FNO2d() if args.variant == "fno" else UNet()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.05)

    it = iter(tr_loader)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tr_loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        pred = model(obs_image(y)).clamp(-1, 1)
        loss_sup = F.mse_loss(pred, x)
        loss_phys = torch.tensor(0.0, device=device)
        if args.variant == "pinn_unet" and step % args.physics_every == 0:
            loss_phys = per_sample_misfit(fwm_call(fwm, pred, args), y).mean()
        loss = loss_sup + args.physics_weight * loss_phys
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == args.steps:
            metrics = evaluate(model, va_loader, fwm, args, device, min(args.n_eval, args.n_val))
            row = {
                "step": step,
                "loss_sup": float(loss_sup.detach().cpu()),
                "loss_phys": float(loss_phys.detach().cpu()),
                "elapsed": time.time() - t0,
                **metrics,
            }
            print(json.dumps(row), flush=True)

    clean_args = {}
    for k, v in vars(args).items():
        if isinstance(v, np.ndarray):
            clean_args[k] = v.tolist()
        else:
            clean_args[k] = v
    final = evaluate(model, va_loader, fwm, args, device, min(args.n_eval, args.n_val))
    final.update({"variant": args.variant, "elapsed_sec": time.time() - t0, "args": clean_args})
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(run_dir, "final.pt"))
    with open(os.path.join(run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
