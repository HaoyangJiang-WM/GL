#!/usr/bin/env python3
"""
run_ddpm_baselines_fwi.py — DDPM-based inverse problem baselines for FWI
=========================================================================
All three share the same unconditional DDPM, differ only at inference.

Variants (--variant):
  dps       DPS  — gradient in xₜ space, chain through Tweedie (Chung 2022)
  mcg       MCG  — gradient in x̂₀ space, then re-noise  (Chung 2022)
  resample  SMC  — K=4 particles, importance-weight resample (Wu 2024)

Setting: 20 DDIM steps, same FWV FVB data as A-LSD experiments.
Model: simple DDPM UNet ~15M params (no conditioning).
"""

import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────── helpers ─────────────────────────────────────────────
def seed_all(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def log_t(x): return torch.log1p(torch.abs(x)) * torch.sign(x)
def seismic_norm(s):
    lo = log_t(torch.tensor(-30., device=s.device, dtype=s.dtype))
    hi = log_t(torch.tensor( 60., device=s.device, dtype=s.dtype))
    return (log_t(s) - lo) / (hi - lo + 1e-6) * 2. - 1.
def to_raw(x): return (x.clamp(-1.2, 1.2) + 1.) * 1500. + 1500.
def pad_v(v, n): return F.pad(v, (n,)*4, mode="replicate")
def fwm_call(fwm, x, a):
    return fwm(pad_v(to_raw(x), a.nbc), a.nbc, a.dx, a.nt, a.dt,
               a.freq, a.sx, a.sz, a.gx, a.gz, a.sampling_rate)

def load_cache(d, n):
    vf = sorted(glob.glob(os.path.join(d,"vel_*.pt")))
    sf = sorted(glob.glob(os.path.join(d,"seis_*.pt")))
    vs, ss = [], []
    for v, s in zip(vf, sf):
        vs.append(torch.load(v))
        ss.append(torch.load(s))
        if sum(x.shape[0] for x in vs) >= n: break
    return torch.cat(vs,0)[:n], torch.cat(ss,0)[:n]

def compact(y): return y[:,:,::10,::7].flatten(1)

# ─────────────────────── DDPM schedule ───────────────────────────────────────
def make_schedule(T=1000, b0=1e-4, b1=0.02):
    betas = torch.linspace(b0, b1, T)
    abar  = torch.cumprod(1-betas, 0)
    return betas, abar

# ─────────────────────── DDPM U-Net ──────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch, td):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.tp = nn.Sequential(nn.SiLU(), nn.Linear(td, ch*2))
    def forward(self, x, te):
        sc, sh = self.tp(te).chunk(2, -1)
        h = self.c1(F.silu(self.n1(x)))
        h = h*(1+sc.view(-1,sc.shape[-1],1,1)) + sh.view(-1,sh.shape[-1],1,1)
        return x + self.c2(F.silu(self.n2(h)))

class UNet(nn.Module):
    def __init__(self, ch=64, td=128, depth=3):
        super().__init__()
        self.td = td
        self.te = nn.Sequential(nn.Linear(td,td*4), nn.SiLU(), nn.Linear(td*4,td))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.downs, self.dcs, self.ups, self.ucs, self.upr = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cc, skips = ch, []
        for _ in range(depth):
            self.downs.append(ResBlock(cc, td)); skips.append(cc)
            self.dcs.append(nn.Conv2d(cc, cc*2, 3, stride=2, padding=1)); cc *= 2
        self.m1 = ResBlock(cc,td); self.m2 = ResBlock(cc,td)
        for sk in reversed(skips):
            self.ucs.append(nn.Conv2d(cc, cc//2, 1)); cc //= 2
            self.upr.append(nn.Conv2d(cc+sk, cc, 1))
            self.ups.append(ResBlock(cc, td))
        self.out = nn.Sequential(nn.GroupNorm(8,cc), nn.SiLU(), nn.Conv2d(cc,1,3,padding=1))

    def _emb(self, t):
        h = self.td//2
        f = torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/h)
        e = torch.cat([torch.sin(t.float()[:,None]*f), torch.cos(t.float()[:,None]*f)], -1)
        return self.te(e)

    def forward(self, x, t):
        te = self._emb(t); h = self.inp(x); sk = []
        for d, dc in zip(self.downs, self.dcs):
            h = d(h, te); sk.append(h); h = dc(h)
        h = self.m2(self.m1(h, te), te)
        for uc, pr, up, s in zip(self.ucs, self.upr, self.ups, reversed(sk)):
            h = uc(h)
            h = F.interpolate(h, size=s.shape[2:], mode="bilinear", align_corners=False)
            h = pr(torch.cat([h,s],1)); h = up(h, te)
        return self.out(h)

# ─────────────────────── DDPM training ───────────────────────────────────────
def train_ddpm(model, train_x, args, device, abar):
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr*0.05)
    T     = len(abar)
    loader = DataLoader(TensorDataset(train_x), args.batch_size, shuffle=True, drop_last=True)
    it = iter(loader); t0 = time.time()
    for step in range(1, args.steps+1):
        try:    (x0,) = next(it)
        except: it = iter(loader); (x0,) = next(it)
        x0 = x0.to(device); b = x0.shape[0]
        ts  = torch.randint(0, T, (b,), device=device)
        eps = torch.randn_like(x0)
        xt  = abar[ts].view(b,1,1,1).sqrt() * x0 + (1-abar[ts].view(b,1,1,1)).sqrt() * eps
        loss = F.mse_loss(model(xt, ts), eps)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            print(json.dumps({"step":step,"loss":float(loss)}), flush=True)
    print(f"DDPM done in {time.time()-t0:.1f}s", flush=True)

# ─────────────────────── inference methods ───────────────────────────────────
def _ddim_step(xt, x0_hat, ab_t, ab_p):
    """Deterministic DDIM step from (xt, x0_hat) → x_{t_prev}."""
    eps_dir = (xt - ab_t.sqrt() * x0_hat) / (1-ab_t).sqrt().clamp_min(1e-8)
    return ab_p.sqrt() * x0_hat + (1-ab_p).sqrt() * eps_dir

@torch.no_grad()
def dps_sample(model, y_seis, fwm, args, device, abar):
    """DPS: gradient in xₜ space (Chung et al. 2022)."""
    T = len(abar); b = y_seis.shape[0]
    ts = list(reversed(range(0, T, T // args.n_steps)))[:args.n_steps]
    xt = torch.randn(b, 1, 70, 70, device=device)
    for i, t in enumerate(ts):
        t_prev = ts[i+1] if i+1<len(ts) else 0
        tv = torch.full((b,), t, device=device, dtype=torch.long)
        ab_t = abar[t].view(1,1,1,1)
        ab_p = abar[t_prev].view(1,1,1,1) if t_prev>0 else torch.zeros(1,1,1,1,device=device)
        with torch.no_grad():
            eps = model(xt, tv)
        x0_hat = (xt - (1-ab_t).sqrt()*eps) / ab_t.sqrt().clamp_min(1e-8)
        xt_ddim = _ddim_step(xt, x0_hat.clamp(-1.5,1.5).detach(), ab_t, ab_p)
        # DPS guidance: grad in xₜ space
        with torch.enable_grad():
            xr = xt.detach().requires_grad_(True)
            x0g = (xr-(1-ab_t).sqrt()*model(xr,tv)) / ab_t.sqrt().clamp_min(1e-8)
            mis = F.mse_loss(seismic_norm(fwm_call(fwm, x0g.clamp(-1.5,1.5), args)), y_seis)
            (g,) = torch.autograd.grad(mis, xr)
        gn = g.flatten(1).norm(dim=1).view(b,1,1,1).clamp_min(1e-8)
        xt = (xt_ddim - args.guidance_zeta * g.detach() / gn).clamp(-1.5,1.5)
    return xt.clamp(-1,1)

@torch.no_grad()
def mcg_sample(model, y_seis, fwm, args, device, abar):
    """MCG: gradient in x̂₀ space + re-noise (Chung et al. 2022)."""
    T = len(abar); b = y_seis.shape[0]
    ts = list(reversed(range(0, T, T // args.n_steps)))[:args.n_steps]
    xt = torch.randn(b, 1, 70, 70, device=device)
    for i, t in enumerate(ts):
        t_prev = ts[i+1] if i+1<len(ts) else 0
        tv = torch.full((b,), t, device=device, dtype=torch.long)
        ab_t = abar[t].view(1,1,1,1)
        ab_p = abar[t_prev].view(1,1,1,1) if t_prev>0 else torch.zeros(1,1,1,1,device=device)
        with torch.no_grad():
            eps = model(xt, tv)
        x0_hat = ((xt-(1-ab_t).sqrt()*eps)/ab_t.sqrt().clamp_min(1e-8)).clamp(-1.5,1.5)
        # MCG: correct x̂₀ with gradient, then re-noise
        with torch.enable_grad():
            xr = x0_hat.detach().requires_grad_(True)
            mis = F.mse_loss(seismic_norm(fwm_call(fwm, xr, args)), y_seis)
            (g,) = torch.autograd.grad(mis, xr)
        gn  = g.flatten(1).norm(dim=1).view(b,1,1,1).clamp_min(1e-8)
        x0_mc = (x0_hat - args.guidance_zeta * g.detach()/gn).clamp(-1.5,1.5)
        # Re-noise x0_mc to current level, then DDIM step
        eps_hat = (xt - ab_t.sqrt()*x0_hat) / (1-ab_t).sqrt().clamp_min(1e-8)
        xt = (ab_p.sqrt()*x0_mc + (1-ab_p).sqrt()*eps_hat.detach()).clamp(-1.5,1.5)
    return xt.clamp(-1,1)

@torch.no_grad()
def resample_sample(model, y_seis, fwm, args, device, abar):
    """SMC resample: K particles, importance-weight every resample_freq steps."""
    K = args.K; T = len(abar); b = y_seis.shape[0]
    ts = list(reversed(range(0, T, T // args.n_steps)))[:args.n_steps]
    xt  = torch.randn(K*b, 1, 70, 70, device=device)
    y_r = y_seis.repeat(K, 1, 1, 1)
    for i, t in enumerate(ts):
        t_prev = ts[i+1] if i+1<len(ts) else 0
        tv  = torch.full((K*b,), t, device=device, dtype=torch.long)
        ab_t = abar[t].view(1,1,1,1)
        ab_p = abar[t_prev].view(1,1,1,1) if t_prev>0 else torch.zeros(1,1,1,1,device=device)
        eps  = model(xt, tv)
        x0_hat = ((xt-(1-ab_t).sqrt()*eps)/ab_t.sqrt().clamp_min(1e-8)).clamp(-1.5,1.5)
        # Resample every `resample_freq` steps
        if i % args.resample_freq == 0:
            seis_p = seismic_norm(fwm_call(fwm, x0_hat, args))
            mf = F.mse_loss(seis_p, y_r, reduction='none').flatten(1).mean(1)  # [K*b]
            mf = mf.view(K, b)
            lw = F.log_softmax(-mf / max(float(1-ab_t), 1e-4), dim=0)  # [K,b]
            w  = lw.exp()
            xt_v = xt.view(K, b, 1, 70, 70)
            for s in range(b):
                idx = torch.multinomial(w[:,s]+1e-8, K, replacement=True)
                xt_v[:,s] = xt_v[idx,s]
        xt = _ddim_step(xt, x0_hat.detach(), ab_t, ab_p).clamp(-1.5,1.5)
    return xt.view(K,b,1,70,70).mean(0).clamp(-1,1)

# ─────────────────────── evaluation ──────────────────────────────────────────
def evaluate_inference(model, val_x, val_seis, fwm, args, device, abar):
    model.eval()
    mses = []
    fn = {"dps": dps_sample, "mcg": mcg_sample, "resample": resample_sample}[args.variant]
    for i in range(0, min(args.n_eval, val_x.shape[0]), args.batch_size):
        x0 = val_x[i:i+args.batch_size].to(device)
        y  = val_seis[i:i+args.batch_size].to(device)
        x0_pred = fn(model, y, fwm, args, device, abar)
        mses.append(F.mse_loss(x0_pred, x0).item())
    model.train()
    return float(np.mean(mses))

# ─────────────────────── main ─────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",       choices=["dps","mcg","resample"], default="mcg")
    p.add_argument("--cache_dir",     default="/data10/fwi_cache/fvb")
    p.add_argument("--out_dir",       default="/data10/alsd_fwi_large")
    p.add_argument("--n_train",       type=int,   default=2000)
    p.add_argument("--n_val",         type=int,   default=400)
    p.add_argument("--steps",         type=int,   default=5000)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_steps",       type=int,   default=20)   # DDIM steps
    p.add_argument("--guidance_zeta", type=float, default=0.15) # DPS/MCG step size
    p.add_argument("--K",             type=int,   default=4)    # resample particles
    p.add_argument("--resample_freq", type=int,   default=4)    # resample every N steps
    p.add_argument("--n_eval",        type=int,   default=32)
    p.add_argument("--nt",            type=int,   default=300)
    p.add_argument("--sampling_rate", type=int,   default=2)
    args = p.parse_args()

    args.nbc=120; args.dx=10; args.dt=1e-3; args.freq=15.; args.sz=10; args.gz=10
    grids=70
    args.sx=np.linspace(0,grids-1,5)*args.dx; args.gx=np.linspace(0,grids-1,grids)*args.dx

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, "/workspace/fdo-fwi/data/fvb")
    from data_gen_f import FWM as fwm

    args.run_dir = os.path.join(args.out_dir, f"ddpm_{args.variant}")
    os.makedirs(args.run_dir, exist_ok=True)

    vel, seis = load_cache(args.cache_dir, args.n_train+args.n_val)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(vel.shape[0], generator=g)
    train_x = vel[perm[:args.n_train]]
    val_x   = vel[perm[args.n_train:]]
    val_s   = seis[perm[args.n_train:]]
    print(f"data: vel={vel.shape} seis={seis.shape}", flush=True)

    betas, abar = make_schedule()
    abar = abar.to(device)

    # Train or load shared DDPM
    shared_ckpt = os.path.join(args.out_dir, "ddpm_shared.pt")
    model = UNet(ch=64, td=128, depth=3).to(device)
    print(f"UNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    if os.path.exists(shared_ckpt):
        model.load_state_dict(torch.load(shared_ckpt, map_location=device))
        print("Loaded shared DDPM checkpoint", flush=True)
    else:
        train_ddpm(model, train_x, args, device, abar)
        torch.save(model.state_dict(), shared_ckpt)
        print(f"Saved shared checkpoint to {shared_ckpt}", flush=True)

    # Evaluate inference variant
    t0 = time.time()
    mse_val = evaluate_inference(model, val_x, val_s, fwm, args, device, abar)
    final = {"variant": args.variant, "mse_inference": mse_val,
             "guidance_zeta": args.guidance_zeta, "n_steps": args.n_steps,
             "K": args.K if args.variant=="resample" else None,
             "elapsed_infer_sec": time.time()-t0}
    with open(os.path.join(args.run_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("FINAL", json.dumps(final), flush=True)

if __name__ == "__main__":
    main()
