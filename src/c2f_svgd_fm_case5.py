#!/usr/bin/env python3
"""FMDA-EKI under FWI acquisition shift.

Use an unconditional FlowMap prior and perform EKI in z_t-space with a
test-time forward operator H. Background can be a supervised NO prediction
or a simple smooth/constant model. This is intentionally inference-only.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def obs_image(seis):
    y = F.interpolate(seis, size=(70, 70), mode="bilinear", align_corners=False)
    return y.reshape(y.shape[0], y.shape[1], 70, 70)


def standardize_obs(y):
    flat = y.flatten(1)
    view_shape = [y.shape[0]] + [1] * (y.dim() - 1)
    mean = flat.mean(dim=1).view(*view_shape)
    std = flat.std(dim=1).view(*view_shape).clamp_min(1e-6)
    return (y - mean) / std


def weighted_normed_misfit(pred_norm, obs_norm, args=None):
    diff2 = (pred_norm - obs_norm).pow(2)
    obs2 = obs_norm.pow(2)
    if args is None or getattr(args, "misfit_mode", "global") == "global":
        return diff2.flatten(1).mean(dim=1)

    weight = torch.ones_like(diff2)
    mute = int(round(float(getattr(args, "direct_mute_frac", 0.0)) * diff2.shape[2]))
    if mute > 0:
        weight[:, :, :mute, :] = 0.0
    late_weight = float(getattr(args, "late_weight", 0.0))
    if late_weight > 0:
        ramp = torch.linspace(0.0, 1.0, diff2.shape[2], device=diff2.device, dtype=diff2.dtype).view(1, 1, -1, 1)
        weight = weight * (1.0 + late_weight * ramp)
    num = (diff2 * weight).flatten(1).sum(dim=1)
    den_w = weight.flatten(1).sum(dim=1).clamp_min(1.0)
    if getattr(args, "misfit_mode", "global") == "relative":
        den = (obs2 * weight).flatten(1).sum(dim=1).clamp_min(1e-6)
        return num / den
    return num / den_w


def per_sample_misfit(pred_norm, obs_norm, args=None):
    """Misfit between normalized predicted and observed seismograms.

    `predict_obs` and `make_observation` already return normalized data.  Do
    not apply `seismic_norm` again here; doing so compresses the prediction a
    second time and can make particle selection/data misfit look artificially
    good.
    """
    return weighted_normed_misfit(pred_norm, obs_norm, args)


def normed_misfit(pred_norm, obs_norm, args=None):
    return weighted_normed_misfit(pred_norm, obs_norm, args)


def parse_float_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def parse_band_list(text):
    if text is None or str(text).strip() == "":
        return [(0.0, 0.18), (0.18, 0.45), (0.45, 1.0)]
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split("-")
        out.append((float(lo), float(hi)))
    return out or [(0.0, 0.18), (0.18, 0.45), (0.45, 1.0)]


def psd_band_misfit(pred_norm, obs_norm, args=None, bands=None, weights=None):
    """Frequency-domain blind score along the time axis.

    This is intentionally simple and inference-only: compare residual power
    in a few frequency bands, normalized by observed power in those bands.
    It is used as an ESD/PSD rank/gate signal, not as a learned model.
    """
    diff = pred_norm - obs_norm
    # Shape convention is [batch, shot/channel, time, receiver].
    r_fft = torch.fft.rfft(diff.float(), dim=2)
    y_fft = torch.fft.rfft(obs_norm.float(), dim=2)
    n_freq = r_fft.shape[2]
    if bands is None:
        bands = parse_band_list(getattr(args, "psd_bands", "0.0-0.18,0.18-0.45,0.45-1.0") if args is not None else "")
    if weights is None:
        weights = parse_float_list(getattr(args, "psd_weights", "") if args is not None else "")
    if len(weights) < len(bands):
        weights = weights + [1.0] * (len(bands) - len(weights))
    total = None
    used = 0.0
    for (lo, hi), w in zip(bands, weights):
        lo_i = max(0, min(n_freq - 1, int(math.floor(lo * (n_freq - 1)))))
        hi_i = max(lo_i + 1, min(n_freq, int(math.ceil(hi * (n_freq - 1)))))
        r_pow = r_fft[:, :, lo_i:hi_i, :].abs().pow(2).flatten(1).sum(dim=1)
        y_pow = y_fft[:, :, lo_i:hi_i, :].abs().pow(2).flatten(1).sum(dim=1).clamp_min(1e-6)
        s = r_pow / y_pow
        total = s * float(w) if total is None else total + s * float(w)
        used += float(w)
    return total / max(used, 1e-6)


def observation_score(pred_norm, obs_norm, args=None):
    base = normed_misfit(pred_norm, obs_norm, args)
    if args is None or str(getattr(args, "psd_score", "none")).lower() == "none":
        return base
    mix = float(getattr(args, "psd_mix", 0.5))
    psd = psd_band_misfit(pred_norm, obs_norm, args).to(base.dtype)
    return (1.0 - mix) * base + mix * psd


def project_obs(y, scale):
    """Simple multiscale observation projection P_l along time/receiver axes."""
    z = y
    if scale > 1:
        z = F.avg_pool2d(z, kernel_size=(scale, 1), stride=(scale, 1), ceil_mode=True)
    # A small receiver-axis smoothing for coarser levels; source axis remains intact.
    if scale >= 4:
        z = F.avg_pool2d(z, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return z


def prepare_obs_for_assim(y, scale, args):
    z = project_obs(y, scale)
    if args.obs_whiten:
        z = standardize_obs(z)
    return z


def parse_int_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def multiscale_score_from_pred(pred_norm, obs_norm, scales, args):
    """Rank candidates with a sum of projected likelihoods.

    A single coarse FWI score can pick the wrong basin.  This implements a
    small continuation/evidence approximation: a candidate must explain the
    observation at several resolutions, not only at one heavily pooled scale.
    """
    if not scales:
        return weighted_normed_misfit(pred_norm, obs_norm, args)
    total = None
    for scale in scales:
        pred_s = prepare_obs_for_assim(pred_norm, scale, args)
        obs_s = prepare_obs_for_assim(obs_norm, scale, args)
        score_s = observation_score(pred_s, obs_s, args)
        total = score_s if total is None else total + score_s
    return total / float(len(scales))


def eki_strength(t_src, args):
    if args.eki_time_power <= 0:
        return args.eki_alpha
    return args.eki_alpha * max(1.0 - float(t_src), 0.0) ** args.eki_time_power


@torch.no_grad()
def ensemble_best_score(flow_model, z, y_obs, loc, t_src, setting, obs_scale, args, device):
    x0 = flow(flow_model, z, t_src, 0.0).clamp(-1, 1)
    pred = prepare_obs_for_assim(predict_obs(x0, loc, setting, args, device), obs_scale, args)
    target = prepare_obs_for_assim(y_obs, obs_scale, args)
    return normed_misfit(pred, target, args).min()


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        groups = min(8, ch)
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * ch))

    def forward(self, x, emb):
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyUNetVel(nn.Module):
    def __init__(self, base=64, emb_dim=192):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.inp = nn.Conv2d(1, base, 3, padding=1)
        self.enc0 = nn.ModuleList([FiLMResBlock(base, emb_dim), FiLMResBlock(base, emb_dim)])
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = nn.ModuleList([FiLMResBlock(base * 2, emb_dim), FiLMResBlock(base * 2, emb_dim)])
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = nn.ModuleList([FiLMResBlock(base * 4, emb_dim), FiLMResBlock(base * 4, emb_dim)])
        self.up1 = nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ModuleList([FiLMResBlock(base * 2, emb_dim), FiLMResBlock(base * 2, emb_dim)])
        self.up0 = nn.Conv2d(base * 2 + base, base, 3, padding=1)
        self.dec0 = nn.ModuleList([FiLMResBlock(base, emb_dim), FiLMResBlock(base, emb_dim)])
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        h0 = self.inp(x)
        for blk in self.enc0:
            h0 = blk(h0, emb)
        h1 = self.down1(h0)
        for blk in self.enc1:
            h1 = blk(h1, emb)
        h2 = self.down2(h1)
        for blk in self.mid:
            h2 = blk(h2, emb)
        u1 = F.interpolate(h2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, h1], dim=1))
        for blk in self.dec1:
            u1 = blk(u1, emb)
        u0 = F.interpolate(u1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, h0], dim=1))
        for blk in self.dec0:
            u0 = blk(u0, emb)
        return self.out(u0)


class OTFM(nn.Module):
    def __init__(self, base=64, emb_dim=192):
        super().__init__()
        self.vel_net = TinyUNetVel(base=base, emb_dim=emb_dim)

    def forward(self, x, t):
        return self.vel_net(x, t)


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
        d2 = self.d2(torch.cat([F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return torch.tanh(self.out(d1))


def load_block(cache_dir, skip, n):
    vf = sorted(glob.glob(os.path.join(cache_dir, "vel_*.pt")))
    sf = sorted(glob.glob(os.path.join(cache_dir, "seis_*.pt")))
    vs, ss = [], []
    for v, s in zip(vf, sf):
        vs.append(torch.load(v, weights_only=False).detach().clone())
        ss.append(torch.load(s, weights_only=False).detach().clone())
        if sum(x.shape[0] for x in vs) >= skip + n:
            break
    return torch.cat(vs, 0)[skip : skip + n], torch.cat(ss, 0)[skip : skip + n]


def load_loc(data_root, skip, n):
    locs = []
    n_files = (skip + n + 499) // 500
    for i in range(1, n_files + 1):
        locs.append(np.load(os.path.join(data_root, "loc", f"loc{i}.npy")).astype(np.float32).reshape(500, 5))
    return np.concatenate(locs, axis=0)[skip : skip + n]


def fwm_batch_f(v_norm, args, device):
    sys.path.insert(0, args.data_root)
    from data_gen_f import FWM

    return FWM(
        pad_v(to_raw(v_norm), args.nbc),
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


def fwm_loc(v_norm, loc, args, device):
    sys.path.insert(0, args.data_root)
    from data_gen_loc import FWM

    rows = []
    for i in range(v_norm.shape[0]):
        rows.append(
            FWM(
                pad_v(to_raw(v_norm[i : i + 1]), args.nbc),
                args.nbc,
                args.dx,
                args.nt,
                args.dt,
                args.freq,
                loc[i if len(loc) > 1 else 0],
                args.sz,
                args.gx,
                args.gz,
                args.sampling_rate,
            )
        )
    return torch.cat(rows, 0)


@torch.no_grad()
def make_observation(x_true, y_f, loc, setting, args, device):
    if setting == "f":
        raw = y_f.to(device)
    else:
        raw = fwm_loc(x_true.to(device), loc, args, device)
    if getattr(args, "score_domain", "norm") == "rawstd":
        return standardize_obs(raw)
    return raw if setting == "f" else seismic_norm(raw)


@torch.no_grad()
def predict_obs(x0, loc, setting, args, device):
    raw = fwm_batch_f(x0, args, device) if setting == "f" else fwm_loc(x0, loc, args, device)
    if getattr(args, "score_domain", "norm") == "rawstd":
        return standardize_obs(raw)
    return seismic_norm(raw)


def predict_obs_grad(x0, loc, setting, args, device):
    raw = fwm_batch_f(x0, args, device) if setting == "f" else fwm_loc(x0, loc, args, device)
    if getattr(args, "score_domain", "norm") == "rawstd":
        return standardize_obs(raw)
    return seismic_norm(raw)


@torch.no_grad()
def flow(flow_model, z, s0, s1):
    return flow_ode(flow_model, z, s0, s1, flow_model.ode_steps).clamp(-1.5, 1.5)


def flow_ode(flow_model, z, s0, s1, steps):
    """Euler integrate OTFM velocity from diffusion time s0 to s1."""
    steps = max(1, int(steps))
    b = z.shape[0]
    x = z
    grid = torch.linspace(float(s0), float(s1), steps + 1, device=z.device)
    for i in range(steps):
        t0 = torch.full((b,), float(grid[i]), device=z.device)
        dt = float(grid[i + 1] - grid[i])
        x = x + dt * flow_model(x, t0)
        x = x.clamp(-1.5, 1.5)
    return x


@torch.no_grad()
def eki_update(flow_model, z, y_obs, loc, t_src, setting, obs_scale, args, device, gamma_scale=1.0):
    n = z.shape[0]
    x0 = flow(flow_model, z, t_src, 0.0).clamp(-1, 1)
    d = prepare_obs_for_assim(predict_obs(x0, loc, setting, args, device), obs_scale, args).flatten(1)
    y = prepare_obs_for_assim(y_obs, obs_scale, args).flatten(1).expand(n, -1)
    zf = z.flatten(1)
    za = zf - zf.mean(0, keepdim=True)
    da = d - d.mean(0, keepdim=True)
    innov = y - d
    mat = da @ da.t() + (n - 1) * args.eki_gamma * gamma_scale * torch.eye(n, device=device)
    rhs = innov @ da.t()
    weights = torch.linalg.solve(mat, rhs.t()).t()
    dz = weights @ za
    if args.trust_rms > 0:
        # Trust region in normalized velocity coordinates: clip each ensemble
        # member's update by per-pixel RMS, not raw high-dimensional L2 size.
        rms = dz.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        dz = dz * (args.trust_rms / rms).clamp(max=1.0)
    if args.trust_ratio > 0:
        d_norm = dz.norm(dim=1, keepdim=True).clamp_min(1e-8)
        z_norm = zf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        dz = dz * (args.trust_ratio * z_norm / d_norm).clamp(max=1.0)
    z_new = (zf + eki_strength(t_src, args) * dz).view_as(z).clamp(-1.5, 1.5)
    if args.eki_inflate != 1.0:
        z_mean = z_new.mean(0, keepdim=True)
        z_new = (z_mean + args.eki_inflate * (z_new - z_mean)).clamp(-1.5, 1.5)
    return z_new, float(innov.pow(2).mean().cpu())


@torch.no_grad()
def systematic_resample(weights):
    n = weights.numel()
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + torch.rand((), device=weights.device)) / n
    cdf = torch.cumsum(weights, dim=0)
    return torch.searchsorted(cdf, positions).clamp(max=n - 1)


@torch.no_grad()
def smc_reweight_resample(flow_model, z, y_obs, loc, t_src, setting, obs_scale, args, device):
    x0 = flow(flow_model, z, t_src, 0.0).clamp(-1, 1)
    pred = prepare_obs_for_assim(predict_obs(x0, loc, setting, args, device), obs_scale, args)
    target = prepare_obs_for_assim(y_obs, obs_scale, args)
    mis = normed_misfit(pred, target, args)
    temp = max(args.smc_temp, 1e-8)
    weights = torch.softmax(-mis / temp, dim=0)
    ess = 1.0 / weights.pow(2).sum().clamp_min(1e-8)
    if ess <= args.smc_ess * z.shape[0]:
        idx = systematic_resample(weights)
        z = z[idx]
        if args.smc_jitter > 0:
            z = (z + args.smc_jitter * torch.randn_like(z)).clamp(-1.5, 1.5)
    return z, float((weights * mis).sum().cpu()), float(ess.cpu())


@torch.no_grad()
def endpoint_mode_resample(flow_model, z, y_obs, loc, t_src, setting, obs_scale, args, device):
    """Select endpoint modes by low-frequency likelihood, then re-inject to z_t.

    This is deliberately different from z-space SMC: the likelihood is used to
    choose clean endpoint candidates x0, and the selected modes are then pushed
    back to diffusion time with fresh noise. That gives the sampler a chance to
    jump between endpoint basins instead of only making local covariance updates
    around the current noisy coordinates.
    """
    n = z.shape[0]
    x0 = flow(flow_model, z, t_src, 0.0).clamp(-1, 1)
    pred = prepare_obs_for_assim(predict_obs(x0, loc, setting, args, device), obs_scale, args)
    target = prepare_obs_for_assim(y_obs, obs_scale, args)
    mis = normed_misfit(pred, target, args)
    if args.mode_select == "topk":
        k = max(1, min(args.mode_topk, n))
        top = torch.topk(-mis, k=k).indices
        reps = torch.randint(0, k, (n,), device=device)
        idx = top[reps]
    else:
        weights = torch.softmax(-mis / max(args.mode_temp, 1e-8), dim=0)
        idx = systematic_resample(weights)
    x_sel = x0[idx]
    if args.mode_noise == "fresh" or float(t_src) <= 1e-6:
        eps = torch.randn_like(x_sel)
    else:
        eps_hat = (z - (1.0 - float(t_src)) * x0) / max(float(t_src), 1e-6)
        eps_sel = eps_hat[idx]
        if args.mode_noise == "selected":
            eps = eps_sel
        elif args.mode_noise == "mix":
            eps = (1.0 - args.mode_noise_mix) * eps_sel + args.mode_noise_mix * torch.randn_like(eps_sel)
        else:
            raise ValueError(f"unknown mode_noise={args.mode_noise}")
    z_reinj = ((1.0 - float(t_src)) * x_sel + float(t_src) * eps).clamp(-1.5, 1.5)
    if args.mode_blend < 1.0:
        z_new = ((1.0 - args.mode_blend) * z[idx] + args.mode_blend * z_reinj).clamp(-1.5, 1.5)
    else:
        z_new = z_reinj
    if args.mode_jitter > 0:
        z_new = (z_new + args.mode_jitter * torch.randn_like(z_new)).clamp(-1.5, 1.5)
    return z_new, float(mis[idx].mean().cpu()), float(mis.min().cpu())


@torch.no_grad()
def fmda_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    z = (1.0 - args.t_start) * x_bg.expand(args.ensemble, -1, -1, -1).to(device)
    z = z + args.t_start * torch.randn_like(z)
    z = z + args.ensemble_spread * torch.randn_like(z)
    t_cur = args.t_start
    last_mis = 0.0
    scales = args.obs_scales
    n_updates = max(1, len(args.times) - 1)
    for update_idx, t_next in enumerate(args.times[1:]):
        obs_scale = scales[min(update_idx, len(scales) - 1)]
        if t_cur <= args.assim_start + 1e-8:
            z_before = z
            score_before = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device) if args.update_gate else None
            for _ in range(args.eki_inner):
                gamma_scale = float(args.eki_inner) if args.eki_mda else 1.0
                z, last_mis = eki_update(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device, gamma_scale)
            if args.update_gate:
                score_after = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device)
                if score_after > score_before * (1.0 + args.update_gate_tol):
                    z = z_before
                    last_mis = float(score_before.cpu())
        z = flow(flow_model, z, t_cur, t_next)
        t_cur = t_next
    x_ens = flow(flow_model, z, t_cur, 0.0).clamp(-1, 1)
    if args.select == "mean":
        x_out = x_ens.mean(0, keepdim=True)
    else:
        pred = predict_obs(x_ens, loc, setting, args, device)
        mis = per_sample_misfit(pred, y_obs, args)
        if args.select == "best":
            x_out = x_ens[int(mis.argmin().item()) : int(mis.argmin().item()) + 1]
        elif args.select == "soft":
            w = torch.softmax(-mis / max(args.select_temp, 1e-6), dim=0).view(-1, 1, 1, 1)
            x_out = (w * x_ens).sum(0, keepdim=True)
        elif args.select == "random":
            j = torch.randint(0, x_ens.shape[0], ()).item()
            x_out = x_ens[j : j + 1]
        else:
            raise ValueError(f"unknown select={args.select}")
    out_mis = per_sample_misfit(predict_obs(x_out, loc, setting, args, device), y_obs, args).mean()
    if args.gate:
        bg_mis = per_sample_misfit(predict_obs(x_bg.to(device), loc, setting, args, device), y_obs, args).mean()
        if float(out_mis.cpu()) > float(bg_mis.cpu()):
            x_out = x_bg.to(device)
            last_mis = float(bg_mis.cpu())
        else:
            last_mis = float(out_mis.cpu())
    else:
        last_mis = float(out_mis.cpu())
    return x_out, last_mis


@torch.no_grad()
def fmda_mode_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    z = (1.0 - args.t_start) * x_bg.expand(args.ensemble, -1, -1, -1).to(device)
    z = z + args.t_start * torch.randn_like(z)
    z = z + args.ensemble_spread * torch.randn_like(z)
    t_cur = args.t_start
    last_mis = 0.0
    scales = args.obs_scales
    for update_idx, t_next in enumerate(args.times[1:]):
        obs_scale = scales[min(update_idx, len(scales) - 1)]
        if t_cur <= args.assim_start + 1e-8:
            z_before = z
            score_before = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device) if args.update_gate else None
            for _ in range(args.mode_repeats):
                z, last_mis, _ = endpoint_mode_resample(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device)
            if args.eki_alpha != 0.0:
                for _ in range(args.eki_inner):
                    gamma_scale = float(args.eki_inner) if args.eki_mda else 1.0
                    z, last_mis = eki_update(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device, gamma_scale)
            if args.update_gate:
                score_after = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device)
                if score_after > score_before * (1.0 + args.update_gate_tol):
                    z = z_before
                    last_mis = float(score_before.cpu())
        z = flow(flow_model, z, t_cur, t_next)
        t_cur = t_next
    x_ens = flow(flow_model, z, t_cur, 0.0).clamp(-1, 1)
    if args.select == "mean":
        x_out = x_ens.mean(0, keepdim=True)
    else:
        pred = predict_obs(x_ens, loc, setting, args, device)
        mis = per_sample_misfit(pred, y_obs, args)
        if args.select == "best":
            x_out = x_ens[int(mis.argmin().item()) : int(mis.argmin().item()) + 1]
        elif args.select == "soft":
            w = torch.softmax(-mis / max(args.select_temp, 1e-6), dim=0).view(-1, 1, 1, 1)
            x_out = (w * x_ens).sum(0, keepdim=True)
        elif args.select == "random":
            j = torch.randint(0, x_ens.shape[0], ()).item()
            x_out = x_ens[j : j + 1]
        else:
            raise ValueError(f"unknown select={args.select}")
    out_mis = per_sample_misfit(predict_obs(x_out, loc, setting, args, device), y_obs, args).mean()
    return x_out, float(out_mis.cpu())


@torch.no_grad()
def fmda_smc_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    z = (1.0 - args.t_start) * x_bg.expand(args.ensemble, -1, -1, -1).to(device)
    z = z + args.t_start * torch.randn_like(z)
    z = z + args.ensemble_spread * torch.randn_like(z)
    t_cur = args.t_start
    last_mis = 0.0
    scales = args.obs_scales
    for update_idx, t_next in enumerate(args.times[1:]):
        obs_scale = scales[min(update_idx, len(scales) - 1)]
        if t_cur <= args.assim_start + 1e-8:
            z_before = z
            score_before = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device) if args.update_gate else None
            for _ in range(args.smc_repeats):
                z, last_mis, _ = smc_reweight_resample(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device)
            if args.eki_alpha != 0.0:
                for _ in range(args.eki_inner):
                    gamma_scale = float(args.eki_inner) if args.eki_mda else 1.0
                    z, last_mis = eki_update(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device, gamma_scale)
            if args.update_gate:
                score_after = ensemble_best_score(flow_model, z, y_obs, loc, t_cur, setting, obs_scale, args, device)
                if score_after > score_before * (1.0 + args.update_gate_tol):
                    z = z_before
                    last_mis = float(score_before.cpu())
        z = flow(flow_model, z, t_cur, t_next)
        t_cur = t_next
    x_ens = flow(flow_model, z, t_cur, 0.0).clamp(-1, 1)
    if args.select == "mean":
        x_out = x_ens.mean(0, keepdim=True)
    else:
        pred = predict_obs(x_ens, loc, setting, args, device)
        mis = per_sample_misfit(pred, y_obs, args)
        if args.select == "best":
            x_out = x_ens[int(mis.argmin().item()) : int(mis.argmin().item()) + 1]
        elif args.select == "soft":
            w = torch.softmax(-mis / max(args.select_temp, 1e-6), dim=0).view(-1, 1, 1, 1)
            x_out = (w * x_ens).sum(0, keepdim=True)
        elif args.select == "random":
            j = torch.randint(0, x_ens.shape[0], ()).item()
            x_out = x_ens[j : j + 1]
        else:
            raise ValueError(f"unknown select={args.select}")
    out_mis = per_sample_misfit(predict_obs(x_out, loc, setting, args, device), y_obs, args).mean()
    if args.gate:
        bg_mis = per_sample_misfit(predict_obs(x_bg.to(device), loc, setting, args, device), y_obs, args).mean()
        if float(out_mis.cpu()) > float(bg_mis.cpu()):
            x_out = x_bg.to(device)
            last_mis = float(bg_mis.cpu())
        else:
            last_mis = float(out_mis.cpu())
    else:
        last_mis = float(out_mis.cpu())
    return x_out, last_mis


def fmda_var_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    z_b = (1.0 - args.t_start) * x_bg.to(device) + args.t_start * torch.randn_like(x_bg.to(device))
    z_b = z_b + args.ensemble_spread * torch.randn_like(z_b)
    z = z_b.detach().clone().requires_grad_(True)
    t_cur = args.t_start
    scales = args.obs_scales
    n_iter = max(1, args.var_steps)
    for update_idx, t_next in enumerate(args.times[1:]):
        obs_scale = scales[min(update_idx, len(scales) - 1)]
        if t_cur > args.assim_start + 1e-8:
            with torch.no_grad():
                z_next = flow(flow_model, z.detach(), t_cur, t_next)
            z_b = flow(flow_model, z_b.detach(), t_cur, t_next)
            z = z_next.detach().clone().requires_grad_(True)
            t_cur = t_next
            continue
        if args.var_opt == "adam":
            opt = torch.optim.Adam([z], lr=args.var_lr)
        elif args.var_opt == "lbfgs":
            opt = torch.optim.LBFGS([z], lr=args.var_lr, max_iter=1, line_search_fn="strong_wolfe")
        else:
            opt = None

        def closure():
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            x0 = flow_ode(flow_model, z, t_cur, 0.0, args.ode_steps).clamp(-1, 1)
            pred = prepare_obs_for_assim(predict_obs_grad(x0, loc, setting, args, device), obs_scale, args)
            target = prepare_obs_for_assim(y_obs, obs_scale, args)
            obs_loss = (pred - target).flatten(1).pow(2).mean()
            bg_loss = (z - z_b).pow(2).mean()
            loss = obs_loss / max(args.var_gamma, 1e-8) + bg_loss / max(args.var_bg, 1e-8)
            loss.backward()
            return loss

        for _ in range(n_iter):
            if args.var_opt == "lbfgs":
                opt.step(closure)
            else:
                if opt is None:
                    z.grad = None
                loss = closure()
                if args.var_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([z], args.var_grad_clip)
                if args.var_opt == "pgd_smooth":
                    with torch.no_grad():
                        grad = z.grad
                        if args.var_smooth_kernel > 1:
                            pad = args.var_smooth_kernel // 2
                            grad = F.avg_pool2d(grad, args.var_smooth_kernel, stride=1, padding=pad)
                        z -= args.var_lr * grad
                else:
                    opt.step()
            with torch.no_grad():
                z.clamp_(-1.5, 1.5)
        with torch.no_grad():
            z_next = flow(flow_model, z.detach(), t_cur, t_next)
        z_b = flow(flow_model, z_b.detach(), t_cur, t_next)
        z = z_next.detach().clone().requires_grad_(True)
        t_cur = t_next
    with torch.no_grad():
        x_out = flow(flow_model, z.detach(), t_cur, 0.0).clamp(-1, 1)
        out_mis = per_sample_misfit(predict_obs(x_out, loc, setting, args, device), y_obs, args).mean()
        if args.gate:
            bg_mis = per_sample_misfit(predict_obs(x_bg.to(device), loc, setting, args, device), y_obs, args).mean()
            if float(out_mis.cpu()) > float(bg_mis.cpu()):
                return x_bg.to(device), float(bg_mis.cpu())
        return x_out, float(out_mis.cpu())


def localmap_refine_anchors(a_init, y_obs, loc, setting, obs_scale, args, device):
    """Clean-space Local MAP assimilation around FlowMap anchors.

    The observation operator is only evaluated on clean physical models.  The
    quadratic anchor term is the local prior/trust region.
    """
    if args.localmap_steps <= 0:
        return a_init.detach().clamp(-1, 1)
    a_ref = a_init.detach().clamp(-1, 1)
    a = a_ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=args.localmap_lr)
    target = prepare_obs_for_assim(y_obs, obs_scale, args)
    before_mis = None
    if args.update_gate:
        with torch.no_grad():
            pred0 = prepare_obs_for_assim(predict_obs(a_ref, loc, setting, args, device), obs_scale, args)
            before_mis = weighted_normed_misfit(pred0, target, args)
    for _ in range(args.localmap_steps):
        opt.zero_grad(set_to_none=True)
        pred = prepare_obs_for_assim(predict_obs_grad(a.clamp(-1, 1), loc, setting, args, device), obs_scale, args)
        obs_loss = weighted_normed_misfit(pred, target, args).mean()
        bg_loss = (a - a_ref).pow(2).mean()
        loss = obs_loss / max(args.localmap_gamma, 1e-8) + bg_loss / max(args.localmap_bg, 1e-8)
        loss.backward()
        if args.localmap_smooth_kernel > 1 and a.grad is not None:
            with torch.no_grad():
                pad = args.localmap_smooth_kernel // 2
                a.grad.copy_(F.avg_pool2d(a.grad, args.localmap_smooth_kernel, stride=1, padding=pad))
        if args.localmap_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([a], args.localmap_grad_clip)
        opt.step()
        with torch.no_grad():
            a.clamp_(-1, 1)
    out = a.detach().clamp(-1, 1)
    if args.update_gate and before_mis is not None:
        with torch.no_grad():
            pred1 = prepare_obs_for_assim(predict_obs(out, loc, setting, args, device), obs_scale, args)
            after_mis = weighted_normed_misfit(pred1, target, args)
            keep_new = after_mis <= before_mis * (1.0 + args.update_gate_tol)
            out = torch.where(keep_new.view(-1, 1, 1, 1), out, a_ref)
    return out


def fmda_localmap_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """Few-step FlowMap with clean-anchor Local MAP assimilation.

    Pure-noise starts are supported by using bg_mode=zero: x_bg is ignored
    except for tensor shape.  At each correction time, decode current states to
    clean anchors, assimilate in clean space, then re-flow the corrected anchors
    back to the current FlowMap time.
    """
    z = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    if args.ensemble_spread > 0:
        z = z + args.ensemble_spread * torch.randn_like(z)
    t_cur = 1.0
    scales = args.obs_scales
    times = [float(t) for t in args.times]
    if len(times) == 0 or abs(times[0] - 1.0) > 1e-6:
        times = [1.0] + times
    if abs(times[-1]) > 1e-6:
        times = times + [0.0]
    for update_idx, t_next in enumerate(times[1:]):
        with torch.no_grad():
            z = flow(flow_model, z, t_cur, t_next)
            a_anchor = flow(flow_model, z, t_next, 0.0).clamp(-1, 1)
        obs_scale = scales[min(update_idx, len(scales) - 1)]
        a_refined = localmap_refine_anchors(a_anchor, y_obs, loc, setting, obs_scale, args, device)
        with torch.no_grad():
            if t_next > 1e-8:
                z = flow(flow_model, a_refined, 0.0, t_next)
            else:
                z = a_refined
            t_cur = t_next
    with torch.no_grad():
        x_ens = z.clamp(-1, 1) if abs(t_cur) <= 1e-8 else flow(flow_model, z, t_cur, 0.0).clamp(-1, 1)
        return select_by_misfit(x_ens, y_obs, loc, setting, args, device)


def clean_refine_batch(x_init, y_obs, loc, setting, args, device):
    if args.clean_steps <= 0:
        return x_init.detach()
    x_ref = x_init.detach()
    x = x_ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=args.clean_lr)
    for _ in range(args.clean_steps):
        opt.zero_grad(set_to_none=True)
        pred = prepare_obs_for_assim(predict_obs_grad(x.clamp(-1, 1), loc, setting, args, device), args.clean_scale, args)
        target = prepare_obs_for_assim(y_obs, args.clean_scale, args)
        obs_loss = weighted_normed_misfit(pred, target, args).mean()
        bg_loss = (x - x_ref).pow(2).mean()
        loss = obs_loss + args.clean_bg * bg_loss
        loss.backward()
        if args.clean_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([x], args.clean_grad_clip)
        opt.step()
        with torch.no_grad():
            x.clamp_(-1, 1)
    return x.detach().clamp(-1, 1)


@torch.no_grad()
def smooth_noise_like(x, kernel):
    noise = torch.randn_like(x)
    if kernel > 1:
        pad = kernel // 2
        noise = F.avg_pool2d(noise, kernel, stride=1, padding=pad)
        rms = noise.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8).view(-1, 1, 1, 1)
        noise = noise / rms
    return noise


@torch.no_grad()
def linearized_refine_batch(x_init, y_obs, loc, setting, args, device):
    """Local linearized clean-space update around each selected anchor.

    Gauss-Newton step in model space using randomised Jacobian (p perturbations):
        dx = α · clip_rms(D_c^T (D_c D_c^T + (p-1)γI)^{-1} (y - H(x₀)) X_c, trust_rms)

    Gates (applied per model):
      1. update_gate  : coarse-scale gate at args.lin_scale — fast, per step
      2. lin_full_gate: full-resolution gate (scale=1) — accurate, per step
         With large trust_rms the coarse gate may accept cycle-skip moves;
         lin_full_gate is the definitive check. Both can be active together:
         coarse gate screens cheaply, full gate confirms.

    To make updates genuinely effective (~100 m/s class):
        --lin_trust_rms 0.08 --lin_alpha 0.6 --lin_steps 3 --lin_full_gate
    The full-res gate ensures safety despite the large step.
    """
    if args.lin_steps <= 0:
        return x_init.detach().clamp(-1, 1)

    lin_full_gate = getattr(args, 'lin_full_gate', False)
    gate_tol = float(getattr(args, 'update_gate_tol', 0.0))

    anchors = x_init.detach().clamp(-1, 1)
    target = prepare_obs_for_assim(y_obs, args.lin_scale, args).flatten(1)
    y_full = y_obs  # reference for full-res gate

    out = anchors
    for _ in range(args.lin_steps):
        refined = []
        for b in range(out.shape[0]):
            x0 = out[b : b + 1].detach().clamp(-1, 1)
            pred0 = prepare_obs_for_assim(predict_obs(x0, loc, setting, args, device), args.lin_scale, args).flatten(1)
            before_coarse = weighted_normed_misfit(
                pred0.view(1, *prepare_obs_for_assim(y_obs, args.lin_scale, args).shape[1:]),
                prepare_obs_for_assim(y_obs, args.lin_scale, args), args)

            # Full-res baseline (only if full gate requested — 1 FWM)
            if lin_full_gate:
                pred0_full = predict_obs(x0, loc, setting, args, device)
                before_full = per_sample_misfit(pred0_full, y_full, args)

            # ── Randomised Jacobian estimation ──────────────────────────────
            p = max(2, args.lin_particles)
            eps = smooth_noise_like(x0.expand(p, -1, -1, -1), args.lin_smooth_kernel)
            x_loc = (x0 + args.lin_sigma * eps).clamp(-1, 1)
            x_loc[0:1] = x0
            d_loc = prepare_obs_for_assim(predict_obs(x_loc, loc, setting, args, device), args.lin_scale, args).flatten(1)
            xf = x_loc.flatten(1)
            xc = xf - xf.mean(0, keepdim=True)
            dc = d_loc - d_loc.mean(0, keepdim=True)
            innov = target - pred0
            mat = dc @ dc.t() + (p - 1) * args.lin_gamma * torch.eye(p, device=device)
            rhs = innov @ dc.t()
            coef = torch.linalg.solve(mat, rhs.t()).t()
            dx = (coef @ xc).view_as(x0)

            # Trust-region clip
            if args.lin_trust_rms > 0:
                rms = dx.pow(2).mean().sqrt().clamp_min(1e-8)
                dx = dx * min(1.0, args.lin_trust_rms / float(rms.cpu()))

            x_new = (x0 + args.lin_alpha * dx).clamp(-1, 1)

            # ── Gate 1: coarse-scale gate (fast) ────────────────────────────
            if args.update_gate:
                pred1 = prepare_obs_for_assim(predict_obs(x_new, loc, setting, args, device), args.lin_scale, args)
                after_coarse = weighted_normed_misfit(pred1, prepare_obs_for_assim(y_obs, args.lin_scale, args), args)
                if float(after_coarse.cpu()) > float(before_coarse.cpu()) * (1.0 + gate_tol):
                    x_new = x0

            # ── Gate 2: full-resolution gate (definitive, 1 extra FWM) ──────
            # Critical when large trust_rms is used: coarse gate may accept
            # cycle-skip moves that hurt full-resolution misfit.  This gate
            # directly checks the quantity we care about.
            if lin_full_gate and (x_new is not x0):
                pred1_full = predict_obs(x_new, loc, setting, args, device)
                after_full = per_sample_misfit(pred1_full, y_full, args)
                if float(after_full.cpu()) > float(before_full.cpu()) * (1.0 + gate_tol):
                    x_new = x0

            refined.append(x_new)
        out = torch.cat(refined, 0).clamp(-1, 1)
    return out


@torch.no_grad()
def select_by_misfit(x_ens, y_obs, loc, setting, args, device):
    pred = predict_obs(x_ens, loc, setting, args, device)
    if bool(getattr(args, "robust_select", False)):
        views = [
            ("focused", False, [16]),
            ("focused", False, [8]),
            ("focused", False, [4, 2, 1]),
            ("global", False, [16, 8, 4, 2, 1]),
            ("relative", True, [16, 8, 4]),
        ]
        rank_sum = torch.zeros(x_ens.shape[0], device=device)
        for mode, whiten, scales_v in views:
            a = copy.copy(args)
            a.misfit_mode = mode
            a.obs_whiten = whiten
            s = multiscale_score_from_pred(pred, y_obs, scales_v, a)
            order = torch.argsort(s)
            ranks = torch.empty_like(s)
            ranks[order] = torch.arange(s.numel(), device=device, dtype=s.dtype)
            rank_sum = rank_sum + ranks
        mis = rank_sum / float(len(views))
    else:
        scales = getattr(args, "select_score_scales", [])
        mis = multiscale_score_from_pred(pred, y_obs, scales, args) if scales else per_sample_misfit(pred, y_obs, args)
    if args.select == "mean":
        return x_ens.mean(0, keepdim=True), float(mis.mean().cpu())
    if args.select == "best":
        j = int(mis.argmin().item())
        return x_ens[j : j + 1], float(mis[j].cpu())
    if args.select == "soft":
        w = torch.softmax(-mis / max(args.select_temp, 1e-6), dim=0).view(-1, 1, 1, 1)
        x = (w * x_ens).sum(0, keepdim=True)
        out_mis = per_sample_misfit(predict_obs(x, loc, setting, args, device), y_obs, args).mean()
        return x, float(out_mis.cpu())
    if args.select == "random":
        j = torch.randint(0, x_ens.shape[0], ()).item()
        x = x_ens[j : j + 1]
        return x, float(mis[j].cpu())
    raise ValueError(f"unknown select={args.select}")


@torch.no_grad()
def proposal_scores(x_ens, y_obs, loc, setting, args, device, scale):
    pred = predict_obs(x_ens, loc, setting, args, device)
    scales = getattr(args, "proposal_score_scales", [])
    if scales:
        return multiscale_score_from_pred(pred, y_obs, scales, args)
    target = prepare_obs_for_assim(y_obs, scale, args)
    pred_s = prepare_obs_for_assim(pred, scale, args)
    return normed_misfit(pred_s, target, args)


@torch.no_grad()
def proposal_union_indices(x_ens, y_obs, loc, setting, args, device, k):
    """Return elite indices from several conservative likelihood views.

    FWI scores are nonconvex and a single projected residual often selects a
    wrong basin even when a good raw FlowMap proposal exists.  This keeps the
    method simple: no new network, no NO init, just a small union of posterior
    evidence approximations before the local linearized correction.
    """
    if not args.proposal_union:
        score = proposal_scores(x_ens, y_obs, loc, setting, args, device, args.proposal_scale)
        return torch.topk(-score, k=k).indices

    pred = predict_obs(x_ens, loc, setting, args, device)
    views = []

    # View 1: current user-specified score.
    views.append((args.misfit_mode, bool(args.obs_whiten), args.proposal_score_scales or [args.proposal_scale]))
    # View 2: original focused low-frequency score, good for broad basin.
    views.append(("focused", False, [16]))
    # View 3: slightly finer focused score.
    views.append(("focused", False, [8]))
    # View 4: relative whitened score, good when amplitude dominates.
    views.append(("relative", True, [16]))

    idxs = []
    per = max(1, int(math.ceil(k / len(views))))
    for mode, whiten, scales in views:
        a = copy.copy(args)
        a.misfit_mode = mode
        a.obs_whiten = whiten
        scores = multiscale_score_from_pred(pred, y_obs, scales, a)
        idxs.append(torch.topk(-scores, k=min(per, x_ens.shape[0])).indices)

    # Stable unique while preserving order.
    seen = set()
    merged = []
    for t in torch.cat(idxs).tolist():
        if t not in seen:
            seen.add(t)
            merged.append(t)
        if len(merged) >= k:
            break
    if len(merged) < k:
        base = proposal_scores(x_ens, y_obs, loc, setting, args, device, args.proposal_scale)
        for t in torch.argsort(base).tolist():
            if t not in seen:
                merged.append(t)
                seen.add(t)
            if len(merged) >= k:
                break
    return torch.tensor(merged, device=x_ens.device, dtype=torch.long)


def proposal_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    z = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z, 1.0, 0.0).clamp(-1, 1)
        k = max(1, min(args.proposal_topk, x_all.shape[0]))
        for _ in range(args.proposal_rounds):
            elite_idx = proposal_union_indices(x_all, y_obs, loc, setting, args, device, k)
            x_elite = x_all[elite_idx]
            draw = torch.randint(0, x_elite.shape[0], (args.ensemble,), device=device)
            x_seed = x_elite[draw]
            eps = torch.randn_like(x_seed)
            t = float(args.proposal_reinject_t)
            z = (1.0 - t) * x_seed + t * eps
            if args.proposal_jitter > 0:
                z = z + args.proposal_jitter * torch.randn_like(z)
            x_all = flow(flow_model, z, t, 0.0).clamp(-1, 1)

        idx = proposal_union_indices(x_all, y_obs, loc, setting, args, device, k)
        x_elite = x_all[idx]

    # Save pre-refinement elite candidates for pool fallback
    # Pool fallback guarantees GN refinement can only help, never hurt.
    # Critical for easy cases where the prior already has good models near truth:
    # if GN pushes them to a wrong basin, pre-GN candidates are preserved.
    x_elite_pre = x_elite.clone()

    x_elite = linearized_refine_batch(x_elite, y_obs, loc, setting, args, device)
    x_ref = clean_refine_batch(x_elite, y_obs, loc, setting, args, device)

    if getattr(args, 'proposal_pool_fallback', False):
        # Pool pre-refinement and post-refinement candidates, select best at full resolution
        x_pool = torch.cat([x_elite_pre, x_ref], dim=0)
        return select_by_misfit(x_pool, y_obs, loc, setting, args, device)

    return select_by_misfit(x_ref, y_obs, loc, setting, args, device)


def proposal_ms_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """Multi-scale proposal selection via single-pass frequency continuation.

    Algorithm:
    1. Generate large prior ensemble, run ONE predict_obs for all N samples.
    2. Initial broad selection — two modes:
       a) Sequential (default): progressive narrowing from coarseest scale to finest.
       b) Union (--proposal_ms_union_k > 0): union of top-k from coarse scale AND
          top-k from a medium scale. This robustly handles both hard cases (large
          anomaly → coarse scale reliable) AND easy cases (small anomaly → coarse
          scale may cycle-skip, medium scale is reliable).
    3. Progressive refinement of the union/initial pool using finer scales.
    4. GN refinement (large trust_rms with lin_full_gate for safe large steps).
    5. Pool {pre-GN, post-GN}, select best at full resolution.

    All scale projections reuse ONE forward model pass → zero extra FWM cost.
    """
    ms_scales = getattr(args, 'proposal_ms_scales', [16, 8, 4, 1])
    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', args.proposal_topk * 2), args.ensemble))
    ms_topk_end = max(1, min(args.proposal_topk, ms_topk_start))
    ms_union_k = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_scale = getattr(args, 'proposal_ms_union_scale', 4)
    # Optional N-way union: comma-sep list of scales, each contributing union_k/N candidates.
    # E.g. "16,8,4,2" → 4 views each contributing top-128 from 2048 = ~512 unique initial pool.
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels = len(ms_scales)

    # ── Phase 1: one batch FWM for all prior samples ─────────────────────────
    z = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z, 1.0, 0.0).clamp(-1, 1)
        # Single predict_obs call → reuse for all scales (free multi-scale scoring)
        pred_all = predict_obs(x_all, loc, setting, args, device)

    # ── Phase 2: initial broad selection ─────────────────────────────────────
    with torch.no_grad():
        if ms_union_k > 0:
            # Union mode: N views, each contributing top-k/N candidates.
            # Guarantees the right-basin sample is in the initial pool for both:
            #   - Hard cases (large anomaly): coarse scale=16 view is reliable
            #   - Easy cases (small anomaly): fine scale=4 view is reliable
            if ms_union_views:
                view_scales = ms_union_views   # user-specified N-way union
            else:
                view_scales = [ms_scales[0], ms_union_scale]   # default 2-way union

            n_views = len(view_scales)
            per_view = max(1, ms_union_k // n_views)
            n_total = x_all.shape[0]
            all_view_idx = []
            for vs in view_scales:
                pred_sv = prepare_obs_for_assim(pred_all, vs, args)
                obs_sv  = prepare_obs_for_assim(y_obs, vs, args)
                score_sv = normed_misfit(pred_sv, obs_sv, args)
                k_sv = min(per_view, n_total)
                all_view_idx.append(torch.topk(-score_sv, k=k_sv).indices)

            # Stable unique union preserving view order
            seen = set()
            merged = []
            for idx_t in all_view_idx:
                for t in idx_t.tolist():
                    if t not in seen:
                        seen.add(t)
                        merged.append(t)
            active_idx = torch.tensor(merged, device=x_all.device, dtype=torch.long)
            # Progressive narrowing uses ALL ms_scales so narrowing goes through every
            # frequency (including scale=1 which must come AFTER coarser scales have
            # already narrowed the pool to avoid cycle-skipping at full resolution).
            remaining_scales = ms_scales
            remaining_levels  = n_levels
        else:
            # Sequential mode (original): start from full ensemble
            active_idx = torch.arange(x_all.shape[0], device=device)
            remaining_scales = ms_scales
            remaining_levels  = n_levels

    # ── Phase 3: progressive narrowing of active pool ─────────────────────────
    for lvl, scale in enumerate(remaining_scales):
        frac = lvl / max(remaining_levels - 1, 1)
        k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
        k_sel = min(k_lvl, active_idx.shape[0])

        with torch.no_grad():
            pred_s = prepare_obs_for_assim(pred_all[active_idx], scale, args)
            obs_s  = prepare_obs_for_assim(y_obs, scale, args)
            score_s = normed_misfit(pred_s, obs_s, args)
            top_local = torch.topk(-score_s, k=k_sel).indices
            active_idx = active_idx[top_local]

    # ── Phase 4: GN refinement on final top-k_end candidates ─────────────────
    x_elite = x_all[active_idx]
    x_elite_pre = x_elite.clone()

    x_elite_ref = linearized_refine_batch(x_elite, y_obs, loc, setting, args, device)
    x_ref = clean_refine_batch(x_elite_ref, y_obs, loc, setting, args, device)

    # Pool fallback: GN can only help, never hurt
    x_pool_final = torch.cat([x_elite_pre, x_ref], dim=0)
    return select_by_misfit(x_pool_final, y_obs, loc, setting, args, device)


# ─────────────────────────────────────────────────────────────────────────────
# NEW UPDATE METHOD 1: Per-sample Adam refinement (exact autograd through FWM)
# Replaces linearized GN. Key advantages vs GN:
#   1. Uses EXACT gradient ∇_x‖H(x)-y‖² (no linearization error)
#   2. Auto-scales to signal strength: tiny anomaly → small gradient → small step
#      This is the key fix for easy cases — no trust-radius overshoot!
#   3. Adam adaptive lr: handles geometry of seismic model space well
#   4. Prior regularization (adam_bg) prevents latent-space drift
#   5. 3-5x cheaper than GN (no p=32 perturbations needed)
# Learns only 3 scalars: adam_lr, adam_bg, adam_steps
# ─────────────────────────────────────────────────────────────────────────────

def adam_refine_batch(x_init, y_obs, loc, setting, args, device):
    """Per-sample Adam optimization with exact autograd through the forward model.

    Each candidate is refined independently. The gradient ‖H(x)-y‖² is exact
    (no linearization), so easy cases (small residual) get small steps naturally.

    Parameters
    ----------
    --adam_steps  : steps per candidate (default 8)
    --adam_lr     : Adam lr (default 0.005)
    --adam_bg     : L2 reg weight toward x_init (prevents overfit, default 0.5)
    --adam_scale  : spatial scale for misfit (default 4 = coarse)
    --adam_scale2 : second scale for dual-scale loss (0 = disabled)
    --adam_gate   : revert if full-res misfit worsens (default True)
    """
    n_steps = getattr(args, 'adam_steps', 8)
    lr = float(getattr(args, 'adam_lr', 0.005))
    bg_reg = float(getattr(args, 'adam_bg', 0.5))
    scale = int(getattr(args, 'adam_scale', 4))
    scale2 = int(getattr(args, 'adam_scale2', 0))
    gate = getattr(args, 'adam_gate', True)
    if n_steps <= 0:
        return x_init.detach().clamp(-1, 1)

    target = prepare_obs_for_assim(y_obs, scale, args)
    if scale2 > 0:
        target2 = prepare_obs_for_assim(y_obs, scale2, args)

    results = []
    for k in range(x_init.shape[0]):
        x0 = x_init[k:k+1].detach().clamp(-1, 1)

        # Baseline misfit for gating
        if gate:
            with torch.no_grad():
                pred0_full = predict_obs(x0, loc, setting, args, device)
                before_full = per_sample_misfit(pred0_full, y_obs, args)

        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            raw_pred = predict_obs_grad(x.clamp(-1, 1), loc, setting, args, device)
            pred_s = prepare_obs_for_assim(raw_pred, scale, args)
            obs_loss = weighted_normed_misfit(pred_s, target, args).squeeze()
            if scale2 > 0:
                pred_s2 = prepare_obs_for_assim(raw_pred, scale2, args)
                obs_loss = obs_loss + weighted_normed_misfit(pred_s2, target2, args).squeeze()
            # L2 pull toward initial position (prevents drift for easy cases)
            bg_loss = bg_reg * (x - x0).pow(2).mean()
            loss = obs_loss + bg_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.data.clamp_(-1, 1)

        x_new = x.detach().clamp(-1, 1)

        # Gate: revert if full-res misfit got worse
        if gate:
            with torch.no_grad():
                pred1_full = predict_obs(x_new, loc, setting, args, device)
                after_full = per_sample_misfit(pred1_full, y_obs, args)
                if float(after_full.cpu()) > float(before_full.cpu()):
                    x_new = x0

        results.append(x_new)

    return torch.cat(results, 0).clamp(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# NEW UPDATE METHOD 2: Ensemble Kalman Inversion (EKI) update
# Replaces linearized GN. Key advantages:
#   1. Uses ENSEMBLE covariance from K=128 members (vs single-point linearization)
#   2. First ES-MDA step is FREE: reuses pred_all already computed
#   3. Handles K=128 members simultaneously: cheaper than per-member GN
#   4. No linearization = handles nonlinear forward model naturally
#   5. Only 2 learned scalars: eki_sigma, eki_steps
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eki_refine_batch(x_elite, pred_elite, y_obs, loc, setting, args, device):
    """ES-MDA (Ensemble Smoother with Multiple Data Assimilation) update.

    The first step is FREE because it reuses pred_elite which was already
    computed in the proposal_ms ensemble pass (no extra FWM calls).

    For K=128 with 3 ES-MDA steps: cost = 128*2 = 256 FWM calls
    vs GN with K=32, 3 steps, p=32: cost = 32*3*32 = 3072 FWM calls  → 12x cheaper!

    Parameters
    ----------
    --eki_steps  : ES-MDA steps (default 3)
    --eki_sigma  : base observation noise (default 0.1)
    --eki_k      : ensemble size kept before EKI (default 128)
    """
    n_steps = getattr(args, 'eki_steps', 3)
    base_sigma = float(getattr(args, 'eki_sigma', 0.1))
    gate = getattr(args, 'lin_full_gate', False)
    scale = int(getattr(args, 'lin_scale', 4))

    x_ens = x_elite.clone().float()
    pred_ens = pred_elite.clone().float()   # FREE — already computed upstream

    # ES-MDA: inflate noise by sqrt(n_steps) per step → sum(1/alpha_i) = 1
    sigma_eff = base_sigma * math.sqrt(float(n_steps))

    # Adaptive sigma: for easy cases (small innovation/signal), use LARGER sigma
    # to limit the Kalman gain (C_yy + sigma^2 I)^{-1} and prevent overshoot.
    # ref_rms=0.15 is typical mid-case level; easy cases have innov_rms<<ref_rms.
    # sigma_adapt = base * max(1, ref_rms / innov_rms)  → large sigma for easy cases
    if getattr(args, 'eki_adaptive_sigma', False):
        _obs_s = prepare_obs_for_assim(y_obs, scale, args).float().flatten(1)
        _prd_s = prepare_obs_for_assim(pred_ens, scale, args).float().flatten(1)
        _innov_rms = float((_obs_s - _prd_s.mean(0, keepdim=True)).pow(2).mean().sqrt())
        _ref_rms = float(getattr(args, 'eki_adaptive_ref_rms', 0.15))
        _innov_clamped = max(_innov_rms, 0.01)
        # Larger sigma for easy (weak signal) → limit Kalman gain → prevent overshoot
        _adapt_sigma = base_sigma * max(1.0, _ref_rms / _innov_clamped)
        _adapt_sigma = min(_adapt_sigma, float(getattr(args, 'eki_adaptive_sigma_max', 4.0)) * base_sigma)
        sigma_eff = _adapt_sigma * math.sqrt(float(n_steps))

    # Baseline for gating
    if gate:
        pred_base_full = predict_obs(x_ens, loc, setting, args, device)
        misfit_before = per_sample_misfit(pred_base_full, y_obs, args).float()
        x_ens_original = x_ens.clone()

    obs_scaled = prepare_obs_for_assim(y_obs, scale, args).float().flatten(1)  # (1, M)

    for step in range(n_steps):
        K = x_ens.shape[0]
        pred_scaled = prepare_obs_for_assim(pred_ens, scale, args).float().flatten(1)  # (K, M)

        x_flat = x_ens.reshape(K, -1)          # (K, Dx)
        pred_mean = pred_scaled.mean(0, keepdim=True)
        x_mean = x_flat.mean(0, keepdim=True)
        A_x = x_flat - x_mean                  # (K, Dx) — model anomalies
        A_pred = pred_scaled - pred_mean        # (K, M)  — prediction anomalies

        # Perturbed observations for ES-MDA
        M = pred_scaled.shape[1]
        eps = sigma_eff * torch.randn(K, M, device=device, dtype=x_flat.dtype)
        innov = obs_scaled.expand(K, -1) + eps - pred_scaled  # (K, M)

        # Kalman gain in K×K ensemble form (avoids large D×M matrices):
        #   S = A_pred A_pred^T / (K-1) + sigma_eff^2 * I_K     [K×K]
        #   ΔX = (A_x^T @ S^{-1} @ (A_pred @ innov^T))^T / (K-1)
        S = A_pred @ A_pred.t() / (K - 1) + sigma_eff**2 * torch.eye(K, device=device, dtype=x_flat.dtype)
        V = A_pred @ innov.t()  # (K, K): column j = A_pred @ innov_j

        try:
            L = torch.linalg.cholesky(S + 1e-6 * torch.eye(K, device=device, dtype=S.dtype))
            W = torch.cholesky_solve(V, L)       # (K, K): S^{-1} V
        except Exception:
            W = torch.linalg.lstsq(S, V).solution

        # ΔX.T = A_x.T @ W / (K-1)  → ΔX = (K, Dx)
        delta_x = (A_x.t() @ W / (K - 1)).t()
        x_flat_new = (x_flat + delta_x).clamp(-1, 1)
        x_ens = x_flat_new.reshape_as(x_elite)

        # Re-predict for next step
        if step < n_steps - 1:
            pred_ens = predict_obs(x_ens, loc, setting, args, device).float()

    # Full-gate: revert members that didn't improve
    if gate:
        pred_final = predict_obs(x_ens, loc, setting, args, device)
        misfit_after = per_sample_misfit(pred_final, y_obs, args).float()
        improved = (misfit_after < misfit_before)
        x_out = torch.where(
            improved.view(-1, 1, 1, 1).to(x_ens.device),
            x_ens, x_ens_original
        )
        return x_out.clamp(-1, 1)

    return x_ens.clamp(-1, 1)


def proposal_ms_adam_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """proposal_ms with Adam refinement instead of linearized GN.

    Uses exact autograd through FWM for per-sample Adam optimization.
    Easy cases: small misfit gradient → small Adam step → no overshoot.
    Hard cases: large misfit gradient → large Adam step → fast convergence.
    """
    ms_scales = getattr(args, 'proposal_ms_scales', [16, 8, 4, 1])
    # Keep more candidates for Adam (can use larger pool efficiently)
    adam_k = getattr(args, 'adam_k', 64)
    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', adam_k * 8), args.ensemble))
    ms_topk_end = adam_k
    ms_union_k = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_scale = getattr(args, 'proposal_ms_union_scale', 4)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels = len(ms_scales)

    # Phase 1: single FWM pass for all prior samples
    z = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

    # Phase 2: union initial selection
    with torch.no_grad():
        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], ms_union_scale]
            n_views = len(view_scales)
            per_view = max(1, ms_union_k // n_views)
            n_total = x_all.shape[0]
            all_view_idx = []
            for vs in view_scales:
                pred_sv = prepare_obs_for_assim(pred_all, vs, args)
                obs_sv  = prepare_obs_for_assim(y_obs, vs, args)
                score_sv = normed_misfit(pred_sv, obs_sv, args)
                k_sv = min(per_view, n_total)
                all_view_idx.append(torch.topk(-score_sv, k=k_sv).indices)
            seen = set(); merged = []
            for idx_t in all_view_idx:
                for t in idx_t.tolist():
                    if t not in seen:
                        seen.add(t); merged.append(t)
            active_idx = torch.tensor(merged, device=x_all.device, dtype=torch.long)
            remaining_scales = ms_scales
            remaining_levels = n_levels
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)
            remaining_scales = ms_scales
            remaining_levels = n_levels

    # Phase 3: progressive narrowing to adam_k candidates
    for lvl, scale in enumerate(remaining_scales):
        frac = lvl / max(remaining_levels - 1, 1)
        k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
        k_sel = min(k_lvl, active_idx.shape[0])
        with torch.no_grad():
            pred_s = prepare_obs_for_assim(pred_all[active_idx], scale, args)
            obs_s  = prepare_obs_for_assim(y_obs, scale, args)
            score_s = normed_misfit(pred_s, obs_s, args)
            top_local = torch.topk(-score_s, k=k_sel).indices
            active_idx = active_idx[top_local]

    # Phase 4: Adam refinement on adam_k candidates
    x_elite = x_all[active_idx]
    x_elite_pre = x_elite.clone()
    x_elite_ref = adam_refine_batch(x_elite, y_obs, loc, setting, args, device)

    # Pool + select best
    x_pool_final = torch.cat([x_elite_pre, x_elite_ref], dim=0)
    return select_by_misfit(x_pool_final, y_obs, loc, setting, args, device)


def proposal_ms_eki_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """proposal_ms with EKI (ES-MDA) refinement instead of linearized GN.

    First EKI step reuses pred_all (FREE). Subsequent steps cost K FWM calls.
    Handles K=128 members simultaneously using ensemble covariance.
    """
    ms_scales = getattr(args, 'proposal_ms_scales', [16, 8, 4, 1])
    eki_k = getattr(args, 'eki_k', 128)
    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', eki_k * 4), args.ensemble))
    ms_topk_end = eki_k
    ms_union_k = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_scale = getattr(args, 'proposal_ms_union_scale', 4)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels = len(ms_scales)

    # Phase 1
    z = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

    # Phase 2: union selection
    with torch.no_grad():
        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], ms_union_scale]
            n_views = len(view_scales)
            per_view = max(1, ms_union_k // n_views)
            n_total = x_all.shape[0]
            all_view_idx = []
            for vs in view_scales:
                pred_sv = prepare_obs_for_assim(pred_all, vs, args)
                obs_sv  = prepare_obs_for_assim(y_obs, vs, args)
                score_sv = normed_misfit(pred_sv, obs_sv, args)
                k_sv = min(per_view, n_total)
                all_view_idx.append(torch.topk(-score_sv, k=k_sv).indices)
            seen = set(); merged = []
            for idx_t in all_view_idx:
                for t in idx_t.tolist():
                    if t not in seen:
                        seen.add(t); merged.append(t)
            active_idx = torch.tensor(merged, device=x_all.device, dtype=torch.long)
            remaining_scales = ms_scales
            remaining_levels = n_levels
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)
            remaining_scales = ms_scales
            remaining_levels = n_levels

    # Phase 3: progressive narrowing to eki_k
    for lvl, scale in enumerate(remaining_scales):
        frac = lvl / max(remaining_levels - 1, 1)
        k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
        k_sel = min(k_lvl, active_idx.shape[0])
        with torch.no_grad():
            pred_s = prepare_obs_for_assim(pred_all[active_idx], scale, args)
            obs_s  = prepare_obs_for_assim(y_obs, scale, args)
            score_s = normed_misfit(pred_s, obs_s, args)
            top_local = torch.topk(-score_s, k=k_sel).indices
            active_idx = active_idx[top_local]

    # Phase 4: EKI refinement — first step is FREE!
    x_elite = x_all[active_idx]
    pred_elite = pred_all[active_idx]   # Already computed — no extra FWM calls!
    x_elite_pre = x_elite.clone()
    x_elite_upd = eki_refine_batch(x_elite, pred_elite, y_obs, loc, setting, args, device)

    x_pool_final = torch.cat([x_elite_pre, x_elite_upd], dim=0)
    x_best = select_by_misfit(x_pool_final, y_obs, loc, setting, args, device)

    # Phase 5 (optional): Post-EKI Adam polish on best candidate.
    # Very cheap: only 1 candidate × post_eki_adam_steps gradient calls.
    # Adam auto-scales step size by gradient magnitude → no overshoot for easy cases.
    n_polish = int(getattr(args, 'post_eki_adam_steps', 0))
    if n_polish > 0:
        # Temporarily override adam args for the polish pass
        _saved = (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate)
        args.adam_steps = n_polish
        args.adam_lr = float(getattr(args, 'post_eki_adam_lr', 0.003))
        args.adam_bg = float(getattr(args, 'post_eki_adam_bg', 0.2))
        args.adam_scale = int(getattr(args, 'lin_scale', 4))
        args.adam_gate = True   # always gate: revert if polish made things worse
        x_polished = adam_refine_batch(x_best.unsqueeze(0), y_obs, loc, setting, args, device)
        (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate) = _saved
        x_best = select_by_misfit(torch.cat([x_best.unsqueeze(0), x_polished], dim=0),
                                   y_obs, loc, setting, args, device)

    return x_best


# ─────────────────────────────────────────────────────────────────────────────
# LG-FMI: Leverage-Gated FlowMap Inversion
# Theory (derived in session):
#   - Endpoint predictor (FM-Tweedie): x̂_1(x_t) = x_t + (1-t)·v_t(x_t)
#   - Leverage Jacobian: J_t^pred = I + (1-t)·∇_x v_t(x_t)
#   - Update efficiency: E(t) = ||J_t^T A^T W r_t||² / ||r_t||²
#   - GN update in x_t space: δx_t = -(J_t^T C_obs^{-1} J_t + λI)^{-1} J_t^T C_obs^{-1} r_t
#   - Trust region: ||δx_t||_endpoint ≤ τ||(1-t)·v_t||
#
# Key advantages over EKI at x_0:
#   1. EKI update at t_mid > 0 stays "on the manifold" — ODE continues from updated x_t
#   2. Endpoint preview scoring: x̂_1 = x_t + (1-t)·v_t is FREE (1 velocity call)
#   3. Leverage gate: skip update for easy cases (low velocity → low leverage)
#      → prevents overshooting tiny anomaly cases (the main failure mode)
# ─────────────────────────────────────────────────────────────────────────────

def proposal_ms_lgfmi_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """LG-FMI: Multi-scale proposal + intermediate-time EKI on ODE manifold.

    Phases:
    1. Sample N=ensemble prior noises z ~ N(0,I), integrate ALL to x_0.
       Compute H(x_all) once — same as proposal_ms_eki (FREE if reused).
    2. 4-way union selection: same as proposal_ms_eki.
    3. Progressive narrowing to lgfmi_k candidates.
    4. LG-FMI stage at t_mid:
       a. Re-integrate selected z's from t=1 to t_mid (cheap, flow model only)
       b. Endpoint preview: x̂_1 = x_t + (1-t)·v_t (1 velocity eval, no FWM)
       c. [Optional] Second filtering by H(x̂_1) to get lgfmi_k_inner candidates
       d. Leverage gate: skip EKI update if ||v_t|| < lgfmi_v_threshold
       e. EKI update in x_t space using H(x̂_1) as ensemble predictions
       f. Continue ODE: updated x_t → x_0 (cheap, flow model only)
    5. Pool pre-EKI x_0 vs post-EKI x_0, select best by full misfit.

    CLI args: --lgfmi_t_mid 0.5 --lgfmi_k 128 --lgfmi_k_inner 0 (0=no second filter)
              --lgfmi_sigma 0.2 --lgfmi_steps 1 --lgfmi_scale 4
              --lgfmi_v_thresh 0.0 (0=no gate, positive=gate on velocity magnitude)
              --lgfmi_gate (if set, gate: revert if post-EKI worse than pre-EKI)
    """
    # ── hyperparameters ──────────────────────────────────────────────────────
    ms_scales     = getattr(args, 'proposal_ms_scales', [16, 8, 4, 1])
    eki_k         = getattr(args, 'eki_k', 128)
    lgfmi_k       = int(getattr(args, 'lgfmi_k', eki_k))
    lgfmi_k_inner = int(getattr(args, 'lgfmi_k_inner', 0))
    lgfmi_t_mid   = float(getattr(args, 'lgfmi_t_mid', 0.5))
    lgfmi_sigma   = float(getattr(args, 'lgfmi_sigma', getattr(args, 'eki_sigma', 0.2)))
    lgfmi_steps   = int(getattr(args, 'lgfmi_steps', 1))
    lgfmi_scale   = int(getattr(args, 'lgfmi_scale', getattr(args, 'lin_scale', 4)))
    lgfmi_v_thresh= float(getattr(args, 'lgfmi_v_thresh', 0.0))
    lgfmi_gate    = bool(getattr(args, 'lgfmi_gate', True))
    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', lgfmi_k * 4), args.ensemble))
    ms_topk_end   = lgfmi_k
    ms_union_k    = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels      = len(ms_scales)

    # ── Phase 1: sample prior, integrate ALL to x_0 ──────────────────────────
    z_all = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all  = flow(flow_model, z_all, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

    # ── Phase 2: union initial selection ──────────────────────────────────────
    with torch.no_grad():
        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], 4]
            n_views = len(view_scales)
            per_view = max(1, ms_union_k // n_views)
            n_total = x_all.shape[0]
            all_view_idx = []
            for vs in view_scales:
                pred_sv = prepare_obs_for_assim(pred_all, vs, args)
                obs_sv  = prepare_obs_for_assim(y_obs, vs, args)
                score_sv = normed_misfit(pred_sv, obs_sv, args)
                k_sv = min(per_view, n_total)
                all_view_idx.append(torch.topk(-score_sv, k=k_sv).indices)
            seen = set(); merged = []
            for idx_t in all_view_idx:
                for t in idx_t.tolist():
                    if t not in seen:
                        seen.add(t); merged.append(t)
            active_idx = torch.tensor(merged, device=x_all.device, dtype=torch.long)
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)

    # ── Phase 3: progressive narrowing to lgfmi_k candidates ─────────────────
    for lvl, scale in enumerate(ms_scales):
        frac = lvl / max(n_levels - 1, 1)
        k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
        k_sel = min(k_lvl, active_idx.shape[0])
        with torch.no_grad():
            pred_s  = prepare_obs_for_assim(pred_all[active_idx], scale, args)
            obs_s   = prepare_obs_for_assim(y_obs, scale, args)
            score_s = normed_misfit(pred_s, obs_s, args)
            top_local  = torch.topk(-score_s, k=k_sel).indices
            active_idx = active_idx[top_local]

    # ── Phase 4: LG-FMI intermediate-time stage ───────────────────────────────
    z_elite = z_all[active_idx]                    # selected prior noises
    x_elite_x0 = x_all[active_idx].clone()        # x_0 before LG-FMI (for gating)

    # 4a. Re-integrate from t=1 → t_mid (flow model only, cheap)
    n_steps_to_mid = max(1, int(round(args.ode_steps * lgfmi_t_mid)))
    n_steps_mid_to_0 = max(1, int(round(args.ode_steps * lgfmi_t_mid)))
    with torch.no_grad():
        x_t = flow_ode(flow_model, z_elite, 1.0, lgfmi_t_mid, n_steps_to_mid).clamp(-1.5, 1.5)

    # 4b. Endpoint preview: x̂_1 = x_t + (1-t)·v_t (1 velocity eval per particle)
    with torch.no_grad():
        K = x_t.shape[0]
        t_vec = torch.full((K,), lgfmi_t_mid, device=device, dtype=x_t.dtype)
        v_t = flow_model(x_t, t_vec)              # velocity field at t_mid
        x_hat_1 = (x_t + (1.0 - lgfmi_t_mid) * v_t).clamp(-1, 1)

    # 4c. Optional second filter by H(x̂_1)  — costs K FWM calls but refines selection
    with torch.no_grad():
        pred_preview = predict_obs(x_hat_1, loc, setting, args, device)
    if lgfmi_k_inner > 0 and lgfmi_k_inner < K:
        ps = prepare_obs_for_assim(pred_preview, lgfmi_scale, args)
        os_ = prepare_obs_for_assim(y_obs, lgfmi_scale, args)
        sc  = normed_misfit(ps, os_, args)
        top_inner = torch.topk(-sc, k=min(lgfmi_k_inner, K)).indices
        x_t          = x_t[top_inner]
        x_hat_1      = x_hat_1[top_inner]
        pred_preview = pred_preview[top_inner]
        x_elite_x0   = x_elite_x0[top_inner]
        v_t          = v_t[top_inner]
        K            = x_t.shape[0]

    # 4d. Leverage gate: velocity magnitude check.
    #     For easy cases (tiny anomaly), ||v_t|| is large everywhere (prior samples
    #     are all noise-like at t_mid, not near truth).  However, SELECTED top-K
    #     candidates that are closest to truth will have relatively smaller residual.
    #     Gate: if innovation rms is very small → skip EKI (already good fit).
    skip_eki = False
    if lgfmi_v_thresh > 0.0:
        _ps = prepare_obs_for_assim(pred_preview, lgfmi_scale, args).float().flatten(1)
        _os = prepare_obs_for_assim(y_obs, lgfmi_scale, args).float().flatten(1)
        _innov_rms = float((_os - _ps.mean(0, keepdim=True)).pow(2).mean().sqrt())
        if _innov_rms < lgfmi_v_thresh:
            skip_eki = True  # already fitting well — don't corrupt with EKI update

    x_t_pre = x_t.clone()
    if not skip_eki:
        # 4e. EKI update in x_t space (same math as eki_refine_batch)
        #     Predictions = H(x̂_1) for each member (endpoint-preview-based ensemble)
        sigma_eff = lgfmi_sigma * math.sqrt(float(lgfmi_steps))

        # Adaptive sigma: easy cases (small innovation) → larger sigma → limit Kalman gain
        if getattr(args, 'eki_adaptive_sigma', False):
            _ps2 = prepare_obs_for_assim(pred_preview, lgfmi_scale, args).float().flatten(1)
            _os2 = prepare_obs_for_assim(y_obs, lgfmi_scale, args).float().flatten(1)
            _innov_rms2 = float((_os2 - _ps2.mean(0, keepdim=True)).pow(2).mean().sqrt())
            _ref_rms2 = float(getattr(args, 'eki_adaptive_ref_rms', 0.15))
            _innov_c = max(_innov_rms2, 0.01)
            _adapt_s  = lgfmi_sigma * max(1.0, _ref_rms2 / _innov_c)
            _adapt_s  = min(_adapt_s, float(getattr(args, 'eki_adaptive_sigma_max', 4.0)) * lgfmi_sigma)
            sigma_eff = _adapt_s * math.sqrt(float(lgfmi_steps))

        x_ens = x_t.clone().float()
        pred_ens = pred_preview.clone().float()

        obs_scaled = prepare_obs_for_assim(y_obs, lgfmi_scale, args).float().flatten(1)

        for step in range(lgfmi_steps):
            K_ = x_ens.shape[0]
            pred_s_k = prepare_obs_for_assim(pred_ens, lgfmi_scale, args).float().flatten(1)
            x_flat   = x_ens.reshape(K_, -1)

            pred_mean = pred_s_k.mean(0, keepdim=True)
            x_mean    = x_flat.mean(0, keepdim=True)
            A_x   = x_flat   - x_mean
            A_pred = pred_s_k - pred_mean

            M   = pred_s_k.shape[1]
            eps = sigma_eff * torch.randn(K_, M, device=device, dtype=x_flat.dtype)
            innov = obs_scaled.expand(K_, -1) + eps - pred_s_k

            S = A_pred @ A_pred.t() / (K_ - 1) + sigma_eff**2 * torch.eye(K_, device=device, dtype=x_flat.dtype)
            V = A_pred @ innov.t()
            try:
                L = torch.linalg.cholesky(S + 1e-6 * torch.eye(K_, device=device, dtype=S.dtype))
                W = torch.cholesky_solve(V, L)
            except Exception:
                W = torch.linalg.lstsq(S, V).solution

            delta_x  = (A_x.t() @ W / (K_ - 1)).t()
            x_flat_new = (x_flat + delta_x).clamp(-1.5, 1.5)
            x_ens = x_flat_new.reshape_as(x_t)

            # Re-predict via endpoint preview for subsequent steps
            if step < lgfmi_steps - 1:
                with torch.no_grad():
                    t_vec2 = torch.full((K_,), lgfmi_t_mid, device=device, dtype=x_ens.dtype)
                    v_t2   = flow_model(x_ens, t_vec2)
                    x_hat2 = (x_ens + (1.0 - lgfmi_t_mid) * v_t2).clamp(-1, 1)
                    pred_ens = predict_obs(x_hat2, loc, setting, args, device).float()

        x_t_upd = x_ens.clamp(-1.5, 1.5)
    else:
        x_t_upd = x_t  # no update (gate triggered)

    # 4f. Continue ODE from updated x_t → x_0
    with torch.no_grad():
        x_final_upd = flow_ode(flow_model, x_t_upd, lgfmi_t_mid, 0.0, n_steps_mid_to_0).clamp(-1, 1)

    # ── Phase 5: gate + final selection ───────────────────────────────────────
    if lgfmi_gate:
        # Pool: x_0 before LG-FMI vs x_0 after LG-FMI
        x_pool = torch.cat([x_elite_x0, x_final_upd], dim=0)
    else:
        x_pool = x_final_upd

    x_best = select_by_misfit(x_pool, y_obs, loc, setting, args, device)

    # Optional post-LG-FMI Adam polish (re-use post_eki_adam_steps arg)
    n_polish = int(getattr(args, 'post_eki_adam_steps', 0))
    if n_polish > 0:
        _saved = (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate)
        args.adam_steps = n_polish
        args.adam_lr    = float(getattr(args, 'post_eki_adam_lr', 0.003))
        args.adam_bg    = float(getattr(args, 'post_eki_adam_bg', 0.2))
        args.adam_scale = lgfmi_scale
        args.adam_gate  = True
        x_polished = adam_refine_batch(x_best.unsqueeze(0), y_obs, loc, setting, args, device)
        (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate) = _saved
        x_best = select_by_misfit(
            torch.cat([x_best.unsqueeze(0), x_polished], dim=0),
            y_obs, loc, setting, args, device)

    return x_best


# ─────────────────────────────────────────────────────────────────────────────
# LG-FMI GRAD: exact-gradient version
#
# Gradient chain:  x_t  →  x̂_1 = x_t+(1-t)v_t(x_t)  →  H(x̂_1)  →  loss
#
#   ∂loss/∂x_t = J_t^{pred,T} · ∂H/∂x̂_1^T · (H(x̂_1)-y)   [exact, via autograd]
#
# where  J_t^{pred} = I + (1-t)·∇_{x_t}v_t   is the endpoint leverage Jacobian.
# PyTorch computes this automatically — no finite differences, no ensemble.
#
# Key properties:
#   • Easy cases (tiny anomaly, small residual) → tiny gradient → NO overshoot
#   • Hard cases → large gradient → aggressively corrected
#   • Trust region in x_t space prevents large manifold violations
#   • After update in x_t, continue ODE → x_0: flow smooths the update
# ─────────────────────────────────────────────────────────────────────────────

def _lgfmi_grad_refine_xt(x_t_k, y_obs, loc, setting, args, device,
                           t_mid, n_steps, lr, trust_rms, bg_reg, scale, gate):
    """Exact-gradient refinement of K candidates in x_t space.

    For each candidate k:
        x_t_var ← x_t_k[k]  (leaf tensor, requires_grad=True)
        for step in range(n_steps):
            v_t       = flow_model(x_t_var, t_mid)      # velocity — differentiable
            x_hat_1   = x_t_var + (1-t_mid)*v_t         # endpoint predictor
            y_pred    = H(x_hat_1)                       # FWM — differentiable!
            loss      = misfit(y_pred, y_obs) + bg_reg*||x_t_var - x_t_init||²
            g         = ∂loss/∂x_t_var                  # exact gradient
            update    = Adam-style momentum step on g
        trust-region clip: ||x_t_new - x_t_init||_rms ≤ trust_rms

    Returns refined x_t tensor, same shape as x_t_k.
    Note: caller must continue ODE from returned x_t to obtain x_0.
    """
    flow_model = args._lgfmi_flow_model  # injected by caller

    results = []
    for k in range(x_t_k.shape[0]):
        x_t_init = x_t_k[k:k+1].detach().clamp(-1.5, 1.5)
        x_t_var  = x_t_init.clone().requires_grad_(True)

        opt = torch.optim.Adam([x_t_var], lr=lr, betas=(0.9, 0.99))
        t_vec = torch.full((1,), t_mid, device=device, dtype=x_t_var.dtype)

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # Endpoint predictor: x̂_1 = x_t + (1-t)*v_t(x_t)  — exact grad path
            v_t    = flow_model(x_t_var, t_vec)
            x_hat_1 = (x_t_var + (1.0 - t_mid) * v_t).clamp(-1.0, 1.0)

            # H(x̂_1) — FWM is differentiable (pure PyTorch tensors)
            pred_s = prepare_obs_for_assim(
                predict_obs_grad(x_hat_1, loc, setting, args, device),
                scale, args)
            obs_s  = prepare_obs_for_assim(y_obs, scale, args)
            loss   = weighted_normed_misfit(pred_s, obs_s, args).squeeze()

            # L2 pull toward initial x_t position (manifold regularisation)
            if bg_reg > 0.0:
                loss = loss + bg_reg * (x_t_var - x_t_init).pow(2).mean()

            loss.backward()
            opt.step()

            with torch.no_grad():
                x_t_var.data.clamp_(-1.5, 1.5)
                # Trust region in x_t space: ||Δx_t||_rms ≤ trust_rms
                if trust_rms > 0.0:
                    delta = x_t_var.data - x_t_init
                    rms   = delta.pow(2).mean().sqrt()
                    if float(rms) > trust_rms:
                        x_t_var.data = x_t_init + delta * (trust_rms / float(rms))

        results.append(x_t_var.detach().clamp(-1.5, 1.5))

    return torch.cat(results, dim=0)


def proposal_ms_lgfmi_grad_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """LG-FMI-Grad: exact-gradient update in intermediate flow-time space.

    Phases 1-3: same union-selection as proposal_ms_eki / lgfmi (EKI version).
    Phase 4 (GRAD):
        a. Re-integrate z_elite: t=1 → t_mid  (cheap, flow model only)
        b. For each of K candidates at x_t:
               Adam update on x_t using ∂H(x̂_1)/∂x_t  (exact autograd)
               x̂_1 = x_t + (1-t_mid)·v_t(x_t)   ← endpoint predictor
        c. Gate: continue ODE from {x_t_before, x_t_after} → x_0; keep better
    Phase 5: pool + final selection.

    CLI args:
        --lgfmi_t_mid        intermediate time (default 0.5)
        --lgfmi_k            candidates selected before grad refine (default 32)
        --lgfmi_grad_steps   Adam steps per candidate (default 5)
        --lgfmi_grad_lr      Adam learning rate in x_t space (default 0.01)
        --lgfmi_grad_trust   trust-region RMS for x_t update (default 0.2)
        --lgfmi_grad_bg      L2 reg toward x_t_init (default 0.05)
        --lgfmi_scale        observation scale for grad misfit (default 4)
        --lgfmi_gate         gate: keep best of pre/post grad (default True)
    """
    # ── hyperparameters ──────────────────────────────────────────────────────
    ms_scales      = getattr(args, 'proposal_ms_scales', [16, 8, 4, 1])
    lgfmi_k        = int(getattr(args, 'lgfmi_k', 32))
    lgfmi_t_mid    = float(getattr(args, 'lgfmi_t_mid', 0.5))
    lgfmi_steps    = int(getattr(args, 'lgfmi_grad_steps', 5))
    lgfmi_lr       = float(getattr(args, 'lgfmi_grad_lr', 0.01))
    lgfmi_trust    = float(getattr(args, 'lgfmi_grad_trust', 0.2))
    lgfmi_bg       = float(getattr(args, 'lgfmi_grad_bg', 0.05))
    lgfmi_scale    = int(getattr(args, 'lgfmi_scale', getattr(args, 'lin_scale', 4)))
    lgfmi_gate     = bool(getattr(args, 'lgfmi_gate', True))
    ms_topk_start  = max(1, min(getattr(args, 'proposal_ms_topk_start', lgfmi_k * 8), args.ensemble))
    ms_topk_end    = lgfmi_k
    ms_union_k     = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels       = len(ms_scales)

    # Inject flow_model reference for use inside _lgfmi_grad_refine_xt
    args._lgfmi_flow_model = flow_model

    # ── Phase 1: sample prior, integrate ALL to x_0 ──────────────────────────
    z_all = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all    = flow(flow_model, z_all, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

    # ── Phase 2: union initial selection ──────────────────────────────────────
    with torch.no_grad():
        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], 4]
            n_views     = len(view_scales)
            per_view    = max(1, ms_union_k // n_views)
            n_total     = x_all.shape[0]
            all_view_idx = []
            for vs in view_scales:
                pred_sv  = prepare_obs_for_assim(pred_all, vs, args)
                obs_sv   = prepare_obs_for_assim(y_obs, vs, args)
                score_sv = normed_misfit(pred_sv, obs_sv, args)
                k_sv     = min(per_view, n_total)
                all_view_idx.append(torch.topk(-score_sv, k=k_sv).indices)
            seen = set(); merged = []
            for idx_t in all_view_idx:
                for t in idx_t.tolist():
                    if t not in seen:
                        seen.add(t); merged.append(t)
            active_idx = torch.tensor(merged, device=x_all.device, dtype=torch.long)
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)

    # ── Phase 3: progressive narrowing to lgfmi_k candidates ─────────────────
    for lvl, scale in enumerate(ms_scales):
        frac   = lvl / max(n_levels - 1, 1)
        k_lvl  = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
        k_sel  = min(k_lvl, active_idx.shape[0])
        with torch.no_grad():
            pred_s  = prepare_obs_for_assim(pred_all[active_idx], scale, args)
            obs_s   = prepare_obs_for_assim(y_obs, scale, args)
            score_s = normed_misfit(pred_s, obs_s, args)
            top_loc = torch.topk(-score_s, k=k_sel).indices
            active_idx = active_idx[top_loc]

    z_elite      = z_all[active_idx]
    x_elite_x0   = x_all[active_idx].clone()   # x_0 before any update (for gate)

    # ── Phase 4a: re-integrate z → x_t_mid ───────────────────────────────────
    n_steps_to_mid  = max(1, int(round(args.ode_steps * lgfmi_t_mid)))
    n_steps_mid_to0 = max(1, int(round(args.ode_steps * lgfmi_t_mid)))
    with torch.no_grad():
        x_t = flow_ode(flow_model, z_elite, 1.0, lgfmi_t_mid, n_steps_to_mid).clamp(-1.5, 1.5)

    # ── Phase 4b: exact-gradient Adam update in x_t space ────────────────────
    x_t_refined = _lgfmi_grad_refine_xt(
        x_t, y_obs, loc, setting, args, device,
        t_mid=lgfmi_t_mid,
        n_steps=lgfmi_steps,
        lr=lgfmi_lr,
        trust_rms=lgfmi_trust,
        bg_reg=lgfmi_bg,
        scale=lgfmi_scale,
        gate=False)   # gating done at x_0 level below

    # ── Phase 4c: continue ODE from refined x_t → x_0 ────────────────────────
    with torch.no_grad():
        x_final_refined = flow_ode(
            flow_model, x_t_refined, lgfmi_t_mid, 0.0, n_steps_mid_to0).clamp(-1, 1)

    # ── Phase 5: gate + pool + final selection ────────────────────────────────
    if lgfmi_gate:
        x_pool = torch.cat([x_elite_x0, x_final_refined], dim=0)
    else:
        x_pool = x_final_refined

    x_best = select_by_misfit(x_pool, y_obs, loc, setting, args, device)

    # Optional post-refinement Adam polish (re-use post_eki_adam_steps)
    n_polish = int(getattr(args, 'post_eki_adam_steps', 0))
    if n_polish > 0:
        _saved = (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate)
        args.adam_steps = n_polish
        args.adam_lr    = float(getattr(args, 'post_eki_adam_lr', 0.003))
        args.adam_bg    = float(getattr(args, 'post_eki_adam_bg', 0.2))
        args.adam_scale = lgfmi_scale
        args.adam_gate  = True
        x_polished = adam_refine_batch(x_best.unsqueeze(0), y_obs, loc, setting, args, device)
        (args.adam_steps, args.adam_lr, args.adam_bg, args.adam_scale, args.adam_gate) = _saved
        x_best = select_by_misfit(
            torch.cat([x_best.unsqueeze(0), x_polished], dim=0),
            y_obs, loc, setting, args, device)

    return x_best


def _parse_float_schedule(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return list(default)
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_int_schedule(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text:
        return list(default)
    return [int(float(x)) for x in text.split(",") if x.strip()]


def _fdps_refine_xt(x_t, y_obs, loc, setting, args, device, t_cur, n_steps, lr,
                    trust_rms, bg_reg, scale, opt_name, repulse):
    """Few-step FlowMap-DPS guidance at a fixed intermediate time.

    This mirrors DPS structurally:
        x_t -> endpoint preview x0_hat -> H(x0_hat) -> loss -> grad wrt x_t.

    The preview uses the rectified-flow one-step endpoint estimator
        x0_hat = x_t + (1 - t) v_theta(x_t, t),
    so the correction stays in flow-time state space instead of directly
    optimizing clean x0.  No learned corrector and no NO initialization.
    """
    flow_model = args._lgfmi_flow_model
    out = []
    K = x_t.shape[0]
    t_vec_full = torch.full((K,), float(t_cur), device=device, dtype=x_t.dtype)

    # Simple active-matter-inspired diversity pressure: while all particles
    # follow their own innovation gradients, weakly push them away from the
    # current swarm mean to reduce premature basin collapse.  Keep it tiny.
    with torch.no_grad():
        swarm_mean = x_t.mean(0, keepdim=True)

    for k in range(K):
        x_init = x_t[k:k+1].detach().clamp(-1.5, 1.5)
        x_var = x_init.clone().requires_grad_(True)
        t_vec = t_vec_full[k:k+1]

        if opt_name == "adam":
            opt = torch.optim.Adam([x_var], lr=lr, betas=(0.9, 0.99))
        elif opt_name == "sgd":
            opt = torch.optim.SGD([x_var], lr=lr)
        elif opt_name == "momentum":
            opt = torch.optim.SGD([x_var], lr=lr, momentum=0.8)
        else:
            raise ValueError(f"unknown fdps_opt={opt_name}")

        for _ in range(max(1, int(n_steps))):
            opt.zero_grad(set_to_none=True)
            v_t = flow_model(x_var, t_vec)
            x_hat0 = (x_var + (1.0 - float(t_cur)) * v_t).clamp(-1.0, 1.0)
            pred_s = prepare_obs_for_assim(
                predict_obs_grad(x_hat0, loc, setting, args, device),
                scale, args)
            obs_s = prepare_obs_for_assim(y_obs, scale, args)
            loss = weighted_normed_misfit(pred_s, obs_s, args).squeeze()
            if bg_reg > 0:
                loss = loss + bg_reg * (x_var - x_init).pow(2).mean()
            if repulse > 0 and K > 1:
                # Maximize distance from swarm mean, bounded by trust clipping.
                loss = loss - repulse * (x_var - swarm_mean).pow(2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x_var.data.clamp_(-1.5, 1.5)
                if trust_rms > 0:
                    delta = x_var.data - x_init
                    rms = delta.pow(2).mean().sqrt()
                    if float(rms) > trust_rms:
                        x_var.data = x_init + delta * (trust_rms / float(rms))

        out.append(x_var.detach().clamp(-1.5, 1.5))

    return torch.cat(out, dim=0)


def _pool_feature(x, size):
    size = max(2, int(size))
    return F.adaptive_avg_pool2d(x, (size, size)).flatten(1)


def _svgd_direction(z, grad_logp, feature_size=12, bandwidth_scale=1.0, repulse=1.0):
    """SVGD direction in flow-time state space using pooled RBF features.

    Attraction is applied in the full z_t tensor, while the kernel/repulsion is
    computed on a low-dimensional pooled representation for numerical stability.
    The repulsive feature-space term is lifted back to image space by bilinear
    interpolation.  This keeps the method cheap enough for differentiable FWI.
    """
    n = z.shape[0]
    if n <= 1:
        return grad_logp

    feat_size = max(2, int(feature_size))
    feat = _pool_feature(z.detach(), feat_size)
    dist2 = torch.cdist(feat, feat).pow(2)
    h = torch.median(dist2.detach()[dist2.detach() > 0]) if (dist2.detach() > 0).any() else dist2.detach().mean()
    h = (h * float(bandwidth_scale)).clamp_min(1e-6)
    k = torch.exp(-dist2 / h)

    g_flat = grad_logp.flatten(1)
    attr = (k.t() @ g_flat / float(n)).view_as(z)

    if repulse <= 0:
        return attr

    # For particle i, sum_j ∇_{z_j} k(z_j, z_i).  Compute in pooled feature
    # space then lift to image space.
    feat_j = feat[:, None, :]
    feat_i = feat[None, :, :]
    rep_feat = (-2.0 / h) * (k[:, :, None] * (feat_j - feat_i)).sum(dim=0) / float(n)
    rep_low = rep_feat.view(n, 1, feat_size, feat_size)
    rep = F.interpolate(rep_low, size=z.shape[-2:], mode="bilinear", align_corners=False)
    return attr + float(repulse) * rep


def _c2f_svgd_refine_xt(x_t, y_obs, loc, setting, args, device, t_cur, n_steps,
                        lr, trust_rms, prior_reg, scale):
    """Coarse-to-fine SVGD FlowMap assimilation at one flow time.

    Memory-efficient: computes gradient for each particle separately (one FWM
    backward at a time), then applies SVGD with kernel repulsion.
    Gradient chain: z_t -> flow_ode -> x_0 -> H(x_0) -> misfit -> dE/dz_t
    """
    flow_model = args._lgfmi_flow_model
    z_prior = x_t.detach().clone().clamp(-1.5, 1.5)
    z = z_prior.clone()
    n_steps = max(1, int(n_steps))
    n_particles = z.shape[0]
    flow_steps = max(1, int(round(args.ode_steps * max(float(t_cur), 1e-3))))
    feature_size = int(getattr(args, "svgd_feature", 12))
    repulse = float(getattr(args, "svgd_repulse", 1.0))
    bandwidth_scale = float(getattr(args, "svgd_bw", 1.0))
    grad_clip = float(getattr(args, "svgd_grad_clip", 0.0))
    obs_s = prepare_obs_for_assim(y_obs, scale, args)
    # single-particle loc (shared across particles for case5)
    loc_i = loc[:1] if loc is not None and len(loc) > 1 else loc

    optim_type = str(getattr(args, "svgd_optim", "svgd"))
    adam_b1 = float(getattr(args, "svgd_adam_b1", 0.9))
    adam_b2 = float(getattr(args, "svgd_adam_b2", 0.999))
    eki_gamma = float(getattr(args, "svgd_eki_gamma", 0.1))
    eki_scale = int(getattr(args, "svgd_eki_scale", 4))
    # Adam state.  The default is station-local Adam, but
    # --svgd_carry_momentum / --svgd_carry_mom preserves the diagonal
    # preconditioner along the FlowMap trajectory instead of restarting it at
    # each t.  This is intentionally conservative: the state is reused only
    # when the particle shape still matches.
    carry_mom = bool(getattr(args, "svgd_carry_momentum", False)) or bool(getattr(args, "svgd_carry_mom", False))
    if carry_mom and hasattr(args, "_svgd_adam_m") and args._svgd_adam_m.shape == z.shape:
        adam_m = args._svgd_adam_m.detach().clone()
        adam_v = args._svgd_adam_v.detach().clone()
        print(f"      [CARRY_MOM t={t_cur:.2f}] reusing Adam state", flush=True)
    else:
        adam_m = torch.zeros_like(z)
        adam_v = torch.zeros_like(z)

    for step_idx in range(n_steps):
        z_det = z.detach()

        # ── EKI: closed-form ensemble Kalman update (no autograd!) ────────────
        if optim_type == "eki":
            with torch.no_grad():
                # 1. FlowMap look-ahead for all particles
                x0_all = flow_ode(flow_model, z_det, float(t_cur), 0.0, flow_steps).clamp(-1, 1)
                # 2. Predict seismograms at coarse scale
                pred_all = prepare_obs_for_assim(
                    predict_obs(x0_all, loc_i, setting, args, device), eki_scale, args)
                obs_c = prepare_obs_for_assim(y_obs, eki_scale, args)
                # 3. Compute residuals (k, obs_dim)
                res = obs_c - pred_all  # (k, ns, nt, ng)
                res_flat = res.flatten(1)  # (k, obs_dim)
                z_flat = z_det.flatten(1)  # (k, z_dim)
                # 4. Ensemble means
                z_mean = z_flat.mean(0)
                r_mean = res_flat.mean(0)
                # 5. Cross-covariance Czr = E[(z-z_mean)(r-r_mean)^T] / (k-1)
                dZ = z_flat - z_mean  # (k, z_dim)
                dR = res_flat - r_mean  # (k, obs_dim)
                # Kalman gain via small-dim solve: K = dZ^T dR (dR^T dR + gamma*I)^{-1}
                k_par = z_det.shape[0]
                Crr = dR.T @ dR / max(k_par - 1, 1)  # (obs_dim, obs_dim) -- too large!
                # Use ensemble-space trick: K y = dZ^T (dR dR^T + gamma*(k-1)*I)^{-1} dR y
                # Solve: (dR dR^T + gamma*(k-1)*I) alpha = dR (r - r_mean)
                # Then dz_i = dZ^T alpha_i
                RRt = dR @ dR.T / max(k_par - 1, 1) + eki_gamma * torch.eye(k_par, device=z_det.device)
                # alpha = RRt^{-1} * (each particle's residual)
                alpha = torch.linalg.solve(RRt, dR)  # (k, k) solve: (k,k)^{-1}*(k,obs) = (k,obs)
                # Correction: dz = dZ^T @ alpha
                dz_flat = (dZ.T @ alpha).T  # (k, z_dim) -- wait, need (k, z_dim)
                # Actually: for each particle i, dz_i = sum_j alpha_{ij} * (z_j - z_mean)
                dz_flat = alpha @ dZ  # (k, z_dim)
                dz = dz_flat.view_as(z_det)
                # Diagnostic
                dz_rms_eki = dz.flatten(1).pow(2).mean(1).sqrt()
                res_rms = res_flat.pow(2).mean(1).sqrt()
                print(f"    [EKI t={t_cur:.2f} s{step_idx+1}/{n_steps}] "
                      f"dz_rms={dz_rms_eki.mean():.5f} res_rms={res_rms.mean():.5f} "
                      f"z_dev={0:.4f}", flush=True)
                z = (z_det + float(lr) * dz).clamp(-1.5, 1.5)
                if trust_rms > 0:
                    delta = z - z_prior
                    rms = delta.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
                    fac = torch.clamp(float(trust_rms) / rms, max=1.0).view(-1, 1, 1, 1)
                    z = (z_prior + delta * fac).clamp(-1.5, 1.5)
            continue  # skip gradient computation below

        # --- Compute gradient for each particle separately (O(1) FWM memory) ---
        grads = []
        data_losses = []
        tmpd_rho   = float(getattr(args, "svgd_tmpd_rho", 0.0))
        ps_k       = max(1, int(getattr(args, "svgd_ps_k", 1)))
        ps_sigma   = float(getattr(args, "svgd_ps_sigma", 0.01))

        for i in range(n_particles):
            z_i = z_det[i:i+1].clone().requires_grad_(True)

            if ps_k > 1:
                # PS+: average gradient over K noisy look-aheads
                g_acc = torch.zeros_like(z_det[i:i+1])
                dl_acc = 0.0
                for _ in range(ps_k):
                    zn = (z_i + ps_sigma * torch.randn_like(z_i)).detach().requires_grad_(True)
                    x0k = flow_ode(flow_model, zn, float(t_cur), 0.0, flow_steps).clamp(-1.0, 1.0)
                    if tmpd_rho > 0:
                        with torch.no_grad():
                            res = obs_s - prepare_obs_for_assim(
                                predict_obs_grad(x0k.detach(), loc_i, setting, args, device), scale, args)
                        x0k = (x0k + tmpd_rho * res.mean()).clamp(-1.0, 1.0)
                    pk = prepare_obs_for_assim(predict_obs_grad(x0k, loc_i, setting, args, device), scale, args)
                    dlk = weighted_normed_misfit(pk, obs_s, args)
                    plk = (zn - z_prior[i:i+1]).pow(2).flatten().mean()
                    ek = dlk.sum() + float(prior_reg) * plk
                    gk = torch.autograd.grad(ek, zn, create_graph=False)[0]
                    g_acc = g_acc + gk.detach() / ps_k
                    dl_acc += float(dlk.sum().item()) / ps_k
                    del zn, x0k, pk, dlk, ek
                    torch.cuda.empty_cache()
                grads.append(g_acc)
                data_losses.append(dl_acc)
            else:
                x0_i = flow_ode(flow_model, z_i, float(t_cur), 0.0, flow_steps).clamp(-1.0, 1.0)
                if tmpd_rho > 0:
                    # TMPD correction in x0 space (approximate J^T*r as scalar broadcast)
                    with torch.no_grad():
                        ypred0 = prepare_obs_for_assim(
                            predict_obs_grad(x0_i.detach(), loc_i, setting, args, device), scale, args)
                        residual = obs_s - ypred0
                    x0_i = (x0_i + tmpd_rho * residual.mean()).clamp(-1.0, 1.0)
                pred_i = prepare_obs_for_assim(
                    predict_obs_grad(x0_i, loc_i, setting, args, device), scale, args)
                data_loss_i = weighted_normed_misfit(pred_i, obs_s, args)
                prior_loss_i = (z_i - z_prior[i:i+1]).pow(2).flatten().mean()
                energy_i = data_loss_i.sum() + float(prior_reg) * prior_loss_i
                g_i = torch.autograd.grad(energy_i, z_i, create_graph=False)[0]
                grads.append(g_i.detach())
                data_losses.append(float(data_loss_i.sum().item()))
                del z_i, x0_i, pred_i, data_loss_i, energy_i
                torch.cuda.empty_cache()
                continue
            del z_i

        grad = torch.cat(grads, dim=0)  # (k, C, H, W)
        grad_logp = -grad

        if grad_clip > 0:
            rms = grad_logp.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
            fac = torch.clamp(float(grad_clip) / rms, max=1.0).view(-1, 1, 1, 1)
            grad_logp = grad_logp * fac

        # Log gradient magnitude for tuning
        grad_rms = grad_logp.flatten(1).pow(2).mean(dim=1).sqrt()
        z_dev = (z_det - z_prior).flatten(1).pow(2).mean(dim=1).sqrt()
        dl_mean = sum(data_losses)/len(data_losses)
        print(f"    [SVGD t={t_cur:.2f} s{step_idx+1}/{n_steps}] "
              f"grad_rms={grad_rms.mean():.5f}(lo={grad_rms.min():.5f},hi={grad_rms.max():.5f}) "
              f"data_loss={dl_mean:.5f} z_dev={z_dev.mean():.4f}(max={z_dev.max():.4f})",
              flush=True)

        # ── Frequency-filter gradient at large t (low-pass for coarse structure) ──
        grad_smooth_flag = bool(getattr(args, "svgd_grad_smooth", False))
        lowfreq_thresh = float(getattr(args, "svgd_lowfreq_t_thresh", 0.3))
        if grad_smooth_flag and t_cur > lowfreq_thresh:
            ks = max(3, int((t_cur - 0.12) / 0.73 * 13)) | 1  # odd, 13@t=0.85→3@t=0.30
            pad = ks // 2
            # Smooth gradient spatially (low-freq only at large t)
            g_4d = grad_logp  # (k, C, H, W)
            grad_logp = F.avg_pool2d(
                F.pad(g_4d, (pad,pad,pad,pad), mode='reflect'),
                ks, stride=1)
            print(f"      [GRAD_SMOOTH t={t_cur:.2f}] kernel={ks}", flush=True)

        adaptive_lr_flag = bool(getattr(args, "svgd_adaptive_lr", False))
        hybrid_rep = float(getattr(args, "svgd_hybrid_rep", 0.0))
        grad_clip_rms_val = float(getattr(args, "svgd_grad_clip_rms", 0.0))
        x0_kernel_flag = bool(getattr(args, "svgd_x0_kernel", False))

        # Raw gradient RMS (before any normalization)
        raw_grad_rms = grad_logp.flatten(1).pow(2).mean(1).sqrt()

        noise_temp_base = float(getattr(args, "svgd_noise_temp", 0.0))
        shared_adam_flag = bool(getattr(args, "svgd_shared_adam", False))

        if optim_type == "adam":
            # Cross-particle shared Adam: use mean grad for v estimation
            if shared_adam_flag and n_particles > 1:
                g_mean = grad_logp.mean(dim=0, keepdim=True)
                adam_m = adam_b1 * adam_m.detach() + (1.0 - adam_b1) * grad_logp
                # Shared v: estimate from all particles' gradient variance
                adam_v = adam_b2 * adam_v.detach() + (1.0 - adam_b2) * g_mean.expand_as(grad_logp).pow(2)
            else:
                adam_m = adam_b1 * adam_m.detach() + (1.0 - adam_b1) * grad_logp
                adam_v = adam_b2 * adam_v.detach() + (1.0 - adam_b2) * grad_logp.pow(2)

            bias1 = 1.0 - adam_b1 ** (step_idx + 1)
            bias2 = 1.0 - adam_b2 ** (step_idx + 1)
            m_hat = adam_m / bias1
            v_hat = adam_v / bias2
            dz = m_hat / (v_hat.sqrt() + 1e-8)

            if adaptive_lr_flag:
                grad_rms_ref = float(getattr(args, '_grad_rms_ref', raw_grad_rms.mean().item()))
                if not hasattr(args, '_grad_rms_ref'):
                    args._grad_rms_ref = float(raw_grad_rms.mean().item())
                lr_scale = (raw_grad_rms / max(args._grad_rms_ref, 1e-8)).clamp(0.1, 5.0)
                dz = dz * lr_scale.view(-1, 1, 1, 1)
                print(f"      [ADAPTIVE_LR] scale={lr_scale.mean():.3f}", flush=True)
            dz_rms = dz.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
            dz = dz / dz_rms.view(-1, 1, 1, 1)

            # Noisy Adam: add temperature-scaled Langevin noise
            if noise_temp_base > 0:
                T = noise_temp_base * float(t_cur) / 0.85  # scale with t
                noise = torch.randn_like(z_det)
                noise_rms = noise.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
                noise_norm = noise / noise_rms.view(-1,1,1,1)
                import math
                noise_scale = math.sqrt(2.0 * float(lr) * T)
                dz = dz + noise_scale / float(lr) * noise_norm  # add relative to lr

            print(f"    [ADAM t={t_cur:.2f} s{step_idx+1}/{n_steps}] "
                  f"grad_rms={raw_grad_rms.mean():.5f} "
                  f"data_loss={sum(data_losses)/len(data_losses):.5f} "
                  f"z_dev={(z_det-z_prior).flatten(1).pow(2).mean(1).sqrt().mean():.4f} "
                  f"z_div={(z_det.flatten(1).std(0)).mean():.4f}"
                  + (f" T={noise_temp_base * float(t_cur) / 0.85:.4f}" if noise_temp_base > 0 else ""),
                  flush=True)
        elif optim_type == "pgd":
            if grad_clip_rms_val > 0:
                rms = grad_logp.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
                fac = (grad_clip_rms_val / rms).clamp(max=1.0).view(-1,1,1,1)
                dz = grad_logp * fac
            else:
                dz_rms = grad_logp.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
                dz = grad_logp / dz_rms.view(-1, 1, 1, 1)
        else:
            # SVGD with optional x0-space kernel
            if x0_kernel_flag:
                with torch.no_grad():
                    x0_for_kernel = flow_ode(flow_model, z_det, float(t_cur), 0.0, flow_steps).clamp(-1,1)
                dz = _svgd_direction(x0_for_kernel, grad_logp,
                                     feature_size=feature_size,
                                     bandwidth_scale=bandwidth_scale, repulse=repulse)
            else:
                dz = _svgd_direction(z_det, grad_logp,
                                     feature_size=feature_size,
                                     bandwidth_scale=bandwidth_scale, repulse=repulse)
            dz_rms = dz.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
            dz = dz / dz_rms.view(-1, 1, 1, 1)

        # Hybrid: add SVGD repulsion on top of Adam/PGD direction
        if hybrid_rep > 0.0 and n_particles > 1:
            rep_only = _svgd_direction(z_det, torch.zeros_like(grad_logp),
                                       feature_size=feature_size,
                                       bandwidth_scale=bandwidth_scale, repulse=1.0)
            rep_rms = rep_only.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
            rep_norm = rep_only / rep_rms.view(-1,1,1,1)
            dz = dz + hybrid_rep * rep_norm
            dz_rms2 = dz.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
            dz = dz / dz_rms2.view(-1,1,1,1)

        z = (z_det + float(lr) * dz).clamp(-1.5, 1.5)

        if trust_rms > 0:
            delta = z - z_prior
            rms = delta.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
            fac = torch.clamp(float(trust_rms) / rms, max=1.0).view(-1, 1, 1, 1)
            z = (z_prior + delta * fac).clamp(-1.5, 1.5)

        # MCG: manifold projection — push each particle's H(Φ(z)) closer to y
        mcg_steps = int(getattr(args, "svgd_mcg_steps", 0))
        mcg_lr    = float(getattr(args, "svgd_mcg_lr", 0.02))
        if mcg_steps > 0:
            with torch.no_grad():
                z_m = z.clone()
                for _m in range(mcg_steps):
                    x0m = flow_ode(flow_model, z_m, float(t_cur), 0.0, flow_steps).clamp(-1, 1)
                    ypm = prepare_obs_for_assim(
                        predict_obs_grad(x0m, loc_i, setting, args, device), scale, args)
                    res_m = (obs_s - ypm).mean()  # scalar residual
                    z_m = (z_m + mcg_lr * res_m).clamp(-1.5, 1.5)
                z = z_m

    if carry_mom and optim_type == "adam":
        args._svgd_adam_m = adam_m.detach().clone()
        args._svgd_adam_v = adam_v.detach().clone()
    return z.detach()


def proposal_ms_svgd_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """C2F-SVGD-FM: coarse-to-fine SVGD assimilation on FlowMap states.

    Raw-noise proposal bank -> multiscale blind ranking -> sparse flow-time
    stations.  At each station, particles are updated in z_t with true
    differentiable H and an RBF-kernel SVGD update plus a trajectory tether.
    No learned corrector and no NO initialization are used.
    """
    ms_scales = getattr(args, 'proposal_ms_scales', [16, 8, 4, 2, 1])
    svgd_k = int(getattr(args, "svgd_k", getattr(args, "fdps_k", 16)))
    t_list = _parse_float_schedule(getattr(args, "svgd_times", None),
                                   _parse_float_schedule(getattr(args, "fdps_times", None),
                                                         [0.85, 0.70, 0.55, 0.40, 0.25, 0.12]))
    scale_list = _parse_int_schedule(getattr(args, "svgd_scales", None),
                                     _parse_int_schedule(getattr(args, "fdps_scales", None),
                                                         [8, 6, 4, 3, 2, 1]))
    steps_list = _parse_int_schedule(getattr(args, "svgd_steps", None), [4, 4, 3, 3, 2, 2])
    lr_list = _parse_float_schedule(getattr(args, "svgd_lr", None), [0.03, 0.025, 0.02, 0.015, 0.01, 0.006])
    prior_list = _parse_float_schedule(getattr(args, "svgd_prior", None), [0.05, 0.08, 0.12, 0.20, 0.35, 0.50])
    trust_list = _parse_float_schedule(getattr(args, "svgd_trust", None), [0.035, 0.03, 0.025, 0.02, 0.015, 0.012])
    for seq in (scale_list, steps_list, lr_list, prior_list, trust_list):
        while len(seq) < len(t_list):
            seq.append(seq[-1])

    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', svgd_k * 8), args.ensemble))
    ms_topk_end = max(1, svgd_k)
    ms_union_k = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels = len(ms_scales)
    args._lgfmi_flow_model = flow_model

    # Raw-noise prior proposal bank with the same robust multiscale narrowing as FDPS.
    z_all = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z_all, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], 4]
            per_view = max(1, ms_union_k // max(1, len(view_scales)))
            seen, merged = set(), []
            for vs in view_scales:
                score = observation_score(
                    prepare_obs_for_assim(pred_all, vs, args),
                    prepare_obs_for_assim(y_obs, vs, args),
                    args)
                for idx in torch.topk(-score, k=min(per_view, x_all.shape[0])).indices.tolist():
                    if idx not in seen:
                        seen.add(idx); merged.append(idx)
            if bool(getattr(args, "psd_split_rank", False)):
                split_bands = parse_band_list(getattr(args, "psd_bands", "0.0-0.18,0.18-0.45,0.45-1.0"))
                split_per = max(1, per_view // 2)
                split_scale = int(getattr(args, "psd_split_scale", view_scales[0]))
                pred_ps = prepare_obs_for_assim(pred_all, split_scale, args)
                obs_ps = prepare_obs_for_assim(y_obs, split_scale, args)
                for band in split_bands:
                    score = psd_band_misfit(pred_ps, obs_ps, args, bands=[band], weights=[1.0])
                    for idx in torch.topk(-score, k=min(split_per, x_all.shape[0])).indices.tolist():
                        if idx not in seen:
                            seen.add(idx); merged.append(idx)
            active_idx = torch.tensor(merged, device=device, dtype=torch.long)
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)

        for lvl, scale in enumerate(ms_scales):
            frac = lvl / max(n_levels - 1, 1)
            k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
            k_sel = min(k_lvl, active_idx.shape[0])
            score = observation_score(
                prepare_obs_for_assim(pred_all[active_idx], scale, args),
                prepare_obs_for_assim(y_obs, scale, args),
                args)
            active_idx = active_idx[torch.topk(-score, k=k_sel).indices]

        z_elite = z_all[active_idx[:svgd_k]]
        x_pre = x_all[active_idx[:svgd_k]].clone()
        print(f"  [PROPOSAL] selected top-{svgd_k} from ensemble={x_all.shape[0]} "
              f"(union narrowed from {args.ensemble} via {len(ms_scales)}-scale filter)", flush=True)

    x_t = None
    t_prev = 1.0
    x_pool = [x_pre]
    for j, t_cur in enumerate(t_list):
        t_cur = float(t_cur)
        steps_to = max(1, int(round(args.ode_steps * abs(t_prev - t_cur))))
        with torch.no_grad():
            if x_t is None:
                x_t = flow_ode(flow_model, z_elite, t_prev, t_cur, steps_to).clamp(-1.5, 1.5)
            else:
                x_t = flow_ode(flow_model, x_t, t_prev, t_cur, steps_to).clamp(-1.5, 1.5)

        before = x_t.detach().clone()
        refined = _c2f_svgd_refine_xt(
            x_t, y_obs, loc, setting, args, device,
            t_cur=t_cur,
            n_steps=steps_list[j],
            lr=lr_list[j],
            trust_rms=trust_list[j],
            prior_reg=prior_list[j],
            scale=scale_list[j])

        with torch.no_grad():
            preview_steps = max(1, int(round(args.ode_steps * max(t_cur, 1e-3))))
            if bool(getattr(args, "svgd_gate", True)):
                gate_scales = _parse_int_schedule(getattr(args, "svgd_gate_scales", ""), [])
                if not gate_scales:
                    gate_scales = [scale_list[j]]
                x_before = flow_ode(flow_model, before, t_cur, 0.0, preview_steps).clamp(-1, 1)
                x_after = flow_ode(flow_model, refined, t_cur, 0.0, preview_steps).clamp(-1, 1)
                pred_before = predict_obs(x_before, loc, setting, args, device)
                pred_after = predict_obs(x_after, loc, setting, args, device)
                score_before = multiscale_score_from_pred(pred_before, y_obs, gate_scales, args)
                score_after = multiscale_score_from_pred(pred_after, y_obs, gate_scales, args)
                tol = float(getattr(args, "svgd_gate_tol", 0.0))
                accept = (score_after <= score_before * (1.0 + tol)).view(-1, 1, 1, 1)
                accept_flat = accept.view(accept.shape[0]).float()
                pct_accept = 100.0 * accept_flat.mean().item()
                sb = score_before.mean().item(); sa = score_after.mean().item()
                print(f"      [GATE t={t_cur:.2f}] accept={pct_accept:.0f}% "
                      f"score_before={sb:.5f} score_after={sa:.5f} "
                      f"improvement={100*(sb-sa)/max(sb,1e-9):.1f}%", flush=True)
                x_t = torch.where(accept, refined, before).detach().clamp(-1.5, 1.5)
                x_pool_entry = torch.where(accept, x_after, x_before).detach().clamp(-1, 1)

                # SIR: resample particles by misfit score
                sir_flag = bool(getattr(args, "svgd_sir", False))
                sir_jitter = float(getattr(args, "svgd_sir_jitter", 0.005))
                if sir_flag and x_t.shape[0] > 1:
                    with torch.no_grad():
                        # Score each particle (lower=better)
                        sir_scores = score_after.detach().float()
                        sir_temp = max(float(getattr(args, "smc_temperature", 0.05)), 1e-6)
                        sir_weights = torch.softmax(-sir_scores / sir_temp, dim=0)
                        # Resample indices
                        sir_idx = torch.multinomial(sir_weights, x_t.shape[0], replacement=True)
                        x_t = x_t[sir_idx] + sir_jitter * torch.randn_like(x_t)
                        x_t = x_t.clamp(-1.5, 1.5)
                        n_unique = len(set(sir_idx.tolist()))
                        print(f"      [SIR t={t_cur:.2f}] resampled {n_unique}/{x_t.shape[0]} unique particles "
                              f"temp={sir_temp} jitter={sir_jitter}", flush=True)

                x_pool.append(x_pool_entry)
            else:
                x_t = refined.detach()
                x_pool.append(flow_ode(flow_model, x_t, t_cur, 0.0, preview_steps).clamp(-1, 1))

        t_prev = t_cur
        # Log inter-station z_t stats
        zt_rms = x_t.flatten(1).pow(2).mean(1).sqrt()
        zt_std = x_t.flatten(1).std(1)
        print(f"    [z_t after station t={t_cur:.2f}] "
              f"rms={zt_rms.mean():.4f} diversity(std)={zt_std.mean():.4f}", flush=True)

    x_final = flow_ode(flow_model, x_t, t_prev, 0.0,
                       max(1, int(round(args.ode_steps * max(t_prev, 1e-3))))).clamp(-1, 1)
    x_pool.append(x_final)

    # Bidirectional C2F-SVGD-FM: blind-select clean candidates from the current
    # pool, push them back to an intermediate FlowMap time, jitter there, then
    # run the remaining C2F stations again.  This is a controlled re-noising
    # basin-escape test that still starts from pure random proposals.
    bidir_rounds = int(getattr(args, "svgd_bidir_rounds", 0))
    bidir_t_list = _parse_float_schedule(getattr(args, "svgd_bidir_times", None), [0.70])
    bidir_k = int(getattr(args, "svgd_bidir_k", svgd_k))
    bidir_jitter = float(getattr(args, "svgd_bidir_jitter", 0.015))
    bidir_temp = max(float(getattr(args, "svgd_bidir_temp", 0.02)), 1e-8)
    bidir_select_scales = _parse_int_schedule(getattr(args, "svgd_bidir_select_scales", ""), [])
    if not bidir_select_scales:
        bidir_select_scales = _parse_int_schedule(getattr(args, "select_score_scales", ""), [16, 8, 4, 2, 1])
    for rr in range(max(0, bidir_rounds)):
        t_re = float(bidir_t_list[rr % len(bidir_t_list)])
        with torch.no_grad():
            pool_all = torch.cat(x_pool, dim=0).clamp(-1, 1)
            pred_pool = predict_obs(pool_all, loc, setting, args, device)
            score_pool = multiscale_score_from_pred(pred_pool, y_obs, bidir_select_scales, args).float()
            k_sel = min(max(1, bidir_k), pool_all.shape[0])
            elite_idx = torch.topk(-score_pool, k=k_sel).indices
            elite_clean = pool_all[elite_idx].detach()
            if elite_clean.shape[0] < svgd_k:
                w = torch.softmax(-score_pool[elite_idx] / bidir_temp, dim=0)
                extra_idx = torch.multinomial(w, svgd_k - elite_clean.shape[0], replacement=True)
                elite_clean = torch.cat([elite_clean, elite_clean[extra_idx]], dim=0)
            elif elite_clean.shape[0] > svgd_k:
                elite_clean = elite_clean[:svgd_k]
            x_t = flow_ode(flow_model, elite_clean, 0.0, t_re,
                           max(1, int(round(args.ode_steps * max(t_re, 1e-3))))).clamp(-1.5, 1.5)
            if bidir_jitter > 0:
                x_t = (x_t + bidir_jitter * torch.randn_like(x_t)).clamp(-1.5, 1.5)
            print(f"  [BIDIR r{rr+1}/{bidir_rounds}] selected k={elite_clean.shape[0]} "
                  f"best_score={score_pool[elite_idx[0]].item():.5f} "
                  f"renoise 0.00->{t_re:.2f} jitter={bidir_jitter}", flush=True)

        if not bool(getattr(args, "svgd_bidir_carry_momentum", False)):
            for attr in ("_svgd_adam_m", "_svgd_adam_v", "_grad_rms_ref"):
                if hasattr(args, attr):
                    delattr(args, attr)

        t_prev = t_re
        round_stations = []
        if not any(abs(float(tt) - t_re) < 1e-6 for tt in t_list):
            nearest = min(range(len(t_list)), key=lambda ii: abs(float(t_list[ii]) - t_re))
            round_stations.append((t_re, nearest))
        for ii, tt in enumerate(t_list):
            if float(tt) <= t_re + 1e-6:
                round_stations.append((float(tt), ii))

        for t_cur, jj in round_stations:
            steps_to = max(1, int(round(args.ode_steps * abs(t_prev - t_cur))))
            if abs(t_prev - t_cur) > 1e-8:
                with torch.no_grad():
                    x_t = flow_ode(flow_model, x_t, t_prev, t_cur, steps_to).clamp(-1.5, 1.5)
            before = x_t.detach().clone()
            refined = _c2f_svgd_refine_xt(
                x_t, y_obs, loc, setting, args, device,
                t_cur=t_cur,
                n_steps=steps_list[jj],
                lr=lr_list[jj],
                trust_rms=trust_list[jj],
                prior_reg=prior_list[jj],
                scale=scale_list[jj])
            with torch.no_grad():
                preview_steps = max(1, int(round(args.ode_steps * max(t_cur, 1e-3))))
                gate_scales = _parse_int_schedule(getattr(args, "svgd_gate_scales", ""), [])
                if not gate_scales:
                    gate_scales = [scale_list[jj]]
                x_before = flow_ode(flow_model, before, t_cur, 0.0, preview_steps).clamp(-1, 1)
                x_after = flow_ode(flow_model, refined, t_cur, 0.0, preview_steps).clamp(-1, 1)
                pred_before = predict_obs(x_before, loc, setting, args, device)
                pred_after = predict_obs(x_after, loc, setting, args, device)
                score_before = multiscale_score_from_pred(pred_before, y_obs, gate_scales, args)
                score_after = multiscale_score_from_pred(pred_after, y_obs, gate_scales, args)
                tol = float(getattr(args, "svgd_gate_tol", 0.0))
                accept = (score_after <= score_before * (1.0 + tol)).view(-1, 1, 1, 1)
                pct_accept = 100.0 * accept.view(accept.shape[0]).float().mean().item()
                sb = score_before.mean().item(); sa = score_after.mean().item()
                print(f"      [BIDIR-GATE r{rr+1} t={t_cur:.2f}] accept={pct_accept:.0f}% "
                      f"score_before={sb:.5f} score_after={sa:.5f} "
                      f"improvement={100*(sb-sa)/max(sb,1e-9):.1f}%", flush=True)
                x_t = torch.where(accept, refined, before).detach().clamp(-1.5, 1.5)
                x_pool.append(torch.where(accept, x_after, x_before).detach().clamp(-1, 1))
                if bool(getattr(args, "svgd_sir", False)) and x_t.shape[0] > 1:
                    sir_temp = max(float(getattr(args, "smc_temperature", 0.05)), 1e-6)
                    sir_weights = torch.softmax(-score_after.detach().float() / sir_temp, dim=0)
                    sir_idx = torch.multinomial(sir_weights, x_t.shape[0], replacement=True)
                    x_t = (x_t[sir_idx] + float(getattr(args, "svgd_sir_jitter", 0.005)) * torch.randn_like(x_t)).clamp(-1.5, 1.5)
                    print(f"      [BIDIR-SIR r{rr+1} t={t_cur:.2f}] unique={len(set(sir_idx.tolist()))}/{x_t.shape[0]}", flush=True)
            t_prev = t_cur
        with torch.no_grad():
            x_final_bidir = flow_ode(flow_model, x_t, t_prev, 0.0,
                                     max(1, int(round(args.ode_steps * max(t_prev, 1e-3))))).clamp(-1, 1)
            x_pool.append(x_final_bidir)
            pred_b = predict_obs(x_final_bidir, loc, setting, args, device)
            mis_b = multiscale_score_from_pred(pred_b, y_obs, bidir_select_scales, args)
            print(f"  [BIDIR r{rr+1}] final blind score mean={mis_b.mean().item():.5f} "
                  f"best={mis_b.min().item():.5f} pool={sum(x.shape[0] for x in x_pool)}", flush=True)

    # ── x0-space Adam post-processing ────────────────────────────────────────
    x0_adam_steps = int(getattr(args, "svgd_x0_adam_steps", 0))
    x0_adam_lr = float(getattr(args, "svgd_x0_adam_lr", 0.005))
    x0_adam_scale = int(getattr(args, "svgd_x0_adam_scale", 4))
    if x0_adam_steps > 0:
        print(f"  [X0_ADAM] post-processing {x_final.shape[0]} particles "
              f"x {x0_adam_steps} steps lr={x0_adam_lr} scale={x0_adam_scale}", flush=True)
        obs_x0 = prepare_obs_for_assim(y_obs, x0_adam_scale, args)
        am = torch.zeros_like(x_final)
        av = torch.zeros_like(x_final)
        b1, b2, eps = 0.9, 0.999, 1e-8
        x_refined = x_final.detach().clone()
        for step in range(x0_adam_steps):
            # Per-particle gradient in x0 space (sequential for memory)
            grads_x0 = []
            dl_vals = []
            for i in range(x_refined.shape[0]):
                xi = x_refined[i:i+1].detach().requires_grad_(True)
                pred_i = prepare_obs_for_assim(
                    predict_obs_grad(xi, loc_i if 'loc_i' in dir() else (loc[:1] if loc is not None and len(loc) > 1 else loc),
                                     setting, args, device), x0_adam_scale, args)
                dl_i = weighted_normed_misfit(pred_i, obs_x0, args).sum()
                gi = torch.autograd.grad(dl_i, xi, create_graph=False)[0]
                grads_x0.append(gi.detach())
                dl_vals.append(float(dl_i.item()))
                del xi, pred_i, dl_i
                torch.cuda.empty_cache()
            g = torch.cat(grads_x0, 0)
            am = b1 * am + (1-b1) * g
            av = b2 * av + (1-b2) * g**2
            mh = am / (1 - b1**(step+1))
            vh = av / (1 - b2**(step+1))
            dz_x0 = mh / (vh.sqrt() + eps)
            dz_rms = dz_x0.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-8)
            dz_x0 = dz_x0 / dz_rms.view(-1,1,1,1)
            x_refined = (x_refined - x0_adam_lr * dz_x0).clamp(-1, 1)
            dl_mean = sum(dl_vals)/len(dl_vals)
            print(f"    [X0_ADAM s{step+1}] data_loss={dl_mean:.5f} "
                  f"grad_rms={g.flatten(1).pow(2).mean(1).sqrt().mean():.5f}", flush=True)
        x_pool.append(x_refined)
        print(f"  [X0_ADAM] done. pool size: {sum(x.shape[0] for x in x_pool)}", flush=True)

    return select_by_misfit(torch.cat(x_pool, dim=0), y_obs, loc, setting, args, device)


def proposal_ms_fdps_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """Clean FlowMap-DPS for one inverse case.

    It keeps the useful v113-style proposal/ranking tricks, then applies true
    DPS-style guidance at several flow times.  No Tweedie, no learned corrector:
    FlowMap supplies the endpoint preview and the guidance variable is x_t.
    """
    ms_scales = getattr(args, 'proposal_ms_scales', [16, 8, 4, 2, 1])
    fdps_k = int(getattr(args, 'fdps_k', getattr(args, 'lgfmi_k', 24)))
    fdps_times = _parse_float_schedule(getattr(args, 'fdps_times', None), [0.4, 0.2, 0.1])
    fdps_scales = _parse_int_schedule(getattr(args, 'fdps_scales', None), [16, 8, 4])
    if len(fdps_scales) < len(fdps_times):
        fdps_scales = fdps_scales + [fdps_scales[-1]] * (len(fdps_times) - len(fdps_scales))
    fdps_steps = int(getattr(args, 'fdps_steps_per_t', 3))
    fdps_lrs = _parse_float_schedule(getattr(args, 'fdps_lr', 0.02), [0.02])
    fdps_trusts = _parse_float_schedule(getattr(args, 'fdps_trust', 0.05), [0.05])
    fdps_bgs = _parse_float_schedule(getattr(args, 'fdps_bg', 0.0), [0.0])
    if len(fdps_lrs) < len(fdps_times):
        fdps_lrs = fdps_lrs + [fdps_lrs[-1]] * (len(fdps_times) - len(fdps_lrs))
    if len(fdps_trusts) < len(fdps_times):
        fdps_trusts = fdps_trusts + [fdps_trusts[-1]] * (len(fdps_times) - len(fdps_trusts))
    if len(fdps_bgs) < len(fdps_times):
        fdps_bgs = fdps_bgs + [fdps_bgs[-1]] * (len(fdps_times) - len(fdps_bgs))
    fdps_opt = str(getattr(args, 'fdps_opt', 'adam')).lower()
    fdps_reselect = bool(getattr(args, 'fdps_reselect', False))
    fdps_retain = int(getattr(args, 'fdps_retain', fdps_k))
    fdps_reinject = float(getattr(args, 'fdps_reinject', 0.0))
    fdps_repulse = float(getattr(args, 'fdps_repulse', 0.0))
    fdps_gate = bool(getattr(args, 'fdps_gate', False))
    fdps_gate_tol = float(getattr(args, 'fdps_gate_tol', 0.0))
    fdps_gate_scales = _parse_int_schedule(getattr(args, 'fdps_gate_scales', ""), [])
    fdps_post_adam_steps = int(getattr(args, 'fdps_post_adam_steps', 0))
    fdps_post_adam_k = int(getattr(args, 'fdps_post_adam_k', min(fdps_k, 8)))

    ms_topk_start = max(1, min(getattr(args, 'proposal_ms_topk_start', fdps_k * 8), args.ensemble))
    ms_topk_end = max(1, fdps_k)
    ms_union_k = getattr(args, 'proposal_ms_union_k', 0)
    ms_union_views_raw = getattr(args, 'proposal_ms_union_views', None)
    ms_union_views = ms_union_views_raw if ms_union_views_raw else None
    n_levels = len(ms_scales)
    args._lgfmi_flow_model = flow_model

    # Prior proposal bank and v113-style multiscale union ranking.
    z_all = torch.randn(args.ensemble, *x_bg.shape[1:], device=device)
    with torch.no_grad():
        x_all = flow(flow_model, z_all, 1.0, 0.0).clamp(-1, 1)
        pred_all = predict_obs(x_all, loc, setting, args, device)

        if ms_union_k > 0:
            view_scales = ms_union_views if ms_union_views else [ms_scales[0], 4]
            per_view = max(1, ms_union_k // max(1, len(view_scales)))
            seen, merged = set(), []
            for vs in view_scales:
                score = normed_misfit(
                    prepare_obs_for_assim(pred_all, vs, args),
                    prepare_obs_for_assim(y_obs, vs, args),
                    args)
                for idx in torch.topk(-score, k=min(per_view, x_all.shape[0])).indices.tolist():
                    if idx not in seen:
                        seen.add(idx); merged.append(idx)
            active_idx = torch.tensor(merged, device=device, dtype=torch.long)
        else:
            active_idx = torch.arange(x_all.shape[0], device=device)

        for lvl, scale in enumerate(ms_scales):
            frac = lvl / max(n_levels - 1, 1)
            k_lvl = max(ms_topk_end, int(ms_topk_start * (1.0 - frac) + ms_topk_end * frac))
            k_sel = min(k_lvl, active_idx.shape[0])
            score = normed_misfit(
                prepare_obs_for_assim(pred_all[active_idx], scale, args),
                prepare_obs_for_assim(y_obs, scale, args),
                args)
            active_idx = active_idx[torch.topk(-score, k=k_sel).indices]

        z_elite = z_all[active_idx[:fdps_k]]
        x_pre = x_all[active_idx[:fdps_k]].clone()

    # FlowMap-DPS stations.  Times are descending from noise to clean.
    x_t = None
    t_prev = 1.0
    x_pool = [x_pre]
    for j, t_cur in enumerate(fdps_times):
        t_cur = float(t_cur)
        if x_t is None:
            steps_to = max(1, int(round(args.ode_steps * abs(t_prev - t_cur))))
            with torch.no_grad():
                x_t = flow_ode(flow_model, z_elite, t_prev, t_cur, steps_to).clamp(-1.5, 1.5)
        else:
            steps_to = max(1, int(round(args.ode_steps * abs(t_prev - t_cur))))
            with torch.no_grad():
                x_t = flow_ode(flow_model, x_t, t_prev, t_cur, steps_to).clamp(-1.5, 1.5)

        x_t_before = x_t.detach().clone()
        x_t_refined = _fdps_refine_xt(
            x_t, y_obs, loc, setting, args, device,
            t_cur=t_cur,
            n_steps=fdps_steps,
            lr=fdps_lrs[j],
            trust_rms=fdps_trusts[j],
            bg_reg=fdps_bgs[j],
            scale=fdps_scales[j],
            opt_name=fdps_opt,
            repulse=fdps_repulse)

        with torch.no_grad():
            preview_steps = max(1, int(round(args.ode_steps * max(t_cur, 1e-3))))
            if fdps_gate:
                x_before_preview = flow_ode(flow_model, x_t_before, t_cur, 0.0,
                                            preview_steps).clamp(-1, 1)
                x_after_preview = flow_ode(flow_model, x_t_refined, t_cur, 0.0,
                                           preview_steps).clamp(-1, 1)
                gate_scales = fdps_gate_scales if fdps_gate_scales else [fdps_scales[j]]
                pred_before = predict_obs(x_before_preview, loc, setting, args, device)
                pred_after = predict_obs(x_after_preview, loc, setting, args, device)
                score_before = multiscale_score_from_pred(pred_before, y_obs, gate_scales, args)
                score_after = multiscale_score_from_pred(pred_after, y_obs, gate_scales, args)
                accept = (score_after <= score_before * (1.0 + fdps_gate_tol)).view(-1, 1, 1, 1)
                x_t = torch.where(accept, x_t_refined, x_t_before).detach().clamp(-1.5, 1.5)
                x_preview = torch.where(accept, x_after_preview, x_before_preview).detach().clamp(-1, 1)
            else:
                x_t = x_t_refined
                x_preview = flow_ode(flow_model, x_t, t_cur, 0.0,
                                     preview_steps).clamp(-1, 1)
            x_pool.append(x_preview)

            if fdps_reselect and x_t.shape[0] > 1:
                pred = predict_obs(x_preview, loc, setting, args, device)
                score = multiscale_score_from_pred(pred, y_obs, getattr(args, "select_score_scales", []), args)
                keep = torch.topk(-score, k=min(fdps_retain, x_t.shape[0])).indices
                x_t = x_t[keep]
                if fdps_reinject > 0 and x_t.shape[0] < fdps_k:
                    # Active-matter-ish reseeding around surviving basins.
                    need = fdps_k - x_t.shape[0]
                    draw = torch.randint(0, x_t.shape[0], (need,), device=device)
                    x_t = torch.cat([x_t, x_t[draw] + fdps_reinject * torch.randn_like(x_t[draw])], dim=0).clamp(-1.5, 1.5)

        t_prev = t_cur
        # Log inter-station z_t stats
        zt_rms = x_t.flatten(1).pow(2).mean(1).sqrt()
        zt_std = x_t.flatten(1).std(1)
        print(f"    [z_t after station t={t_cur:.2f}] "
              f"rms={zt_rms.mean():.4f} diversity(std)={zt_std.mean():.4f}", flush=True)

    x_final = flow_ode(flow_model, x_t, t_prev, 0.0,
                       max(1, int(round(args.ode_steps * max(t_prev, 1e-3))))).clamp(-1, 1)
    x_pool.append(x_final)
    if fdps_post_adam_steps > 0:
        x_candidates = torch.cat(x_pool, dim=0)
        with torch.no_grad():
            pred = predict_obs(x_candidates, loc, setting, args, device)
            score = multiscale_score_from_pred(pred, y_obs, getattr(args, "select_score_scales", []), args)
            keep = torch.topk(-score, k=min(fdps_post_adam_k, x_candidates.shape[0])).indices
            x_for_adam = x_candidates[keep]

        old_vals = {
            "adam_steps": getattr(args, "adam_steps", None),
            "adam_lr": getattr(args, "adam_lr", None),
            "adam_bg": getattr(args, "adam_bg", None),
            "adam_scale": getattr(args, "adam_scale", None),
            "adam_scale2": getattr(args, "adam_scale2", None),
            "adam_gate": getattr(args, "adam_gate", None),
        }
        args.adam_steps = fdps_post_adam_steps
        args.adam_lr = float(getattr(args, "fdps_post_adam_lr", 0.0015))
        args.adam_bg = float(getattr(args, "fdps_post_adam_bg", 0.35))
        args.adam_scale = int(getattr(args, "fdps_post_adam_scale", 4))
        args.adam_scale2 = int(getattr(args, "fdps_post_adam_scale2", 0))
        args.adam_gate = True
        try:
            x_pool.append(adam_refine_batch(x_for_adam, y_obs, loc, setting, args, device))
        finally:
            for key, val in old_vals.items():
                if val is not None:
                    setattr(args, key, val)
    return select_by_misfit(torch.cat(x_pool, dim=0), y_obs, loc, setting, args, device)


def smooth_bg(x):
    return F.interpolate(F.avg_pool2d(x, 10, stride=10), size=(70, 70), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Cascaded multi-scale proposal + linearized refinement
# Theory:
#   1. Basin coverage : mix of random prior draws AND NO-seeded draws
#      NO-seeded: add noise to NO output at t∈(0,1) → re-flow → manifold-consistent
#      Captures two complementary information sources: prior diversity + NO hint
#   2. Frequency continuation via cascaded linearization
#      Coarse scale (16) → medium (4) → fine (1): avoids cycle-skipping
#      Each level does p-ensemble Gauss-Newton: dx = α (J^T J + γI)^{-1} J^T r
#      where J estimated by random perturbations (randomized Jacobian, ENKF limit)
#   3. Trust region + update gate: safe iteration even with nonlinear FWM
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def linearized_refine_at_scale(x_anchors, y_obs, loc, setting, scale, n_steps, n_particles,
                                lin_sigma, lin_alpha, lin_gamma, lin_trust_rms, lin_smooth_kernel,
                                update_gate, update_gate_tol, args, device):
    """Run linearized_refine_batch with temporarily overridden hyperparams."""
    orig = (args.lin_steps, args.lin_particles, args.lin_scale,
            args.lin_sigma, args.lin_alpha, args.lin_gamma,
            args.lin_trust_rms, args.lin_smooth_kernel)
    args.lin_steps        = n_steps
    args.lin_particles    = n_particles
    args.lin_scale        = scale
    args.lin_sigma        = lin_sigma
    args.lin_alpha        = lin_alpha
    args.lin_gamma        = lin_gamma
    args.lin_trust_rms    = lin_trust_rms
    args.lin_smooth_kernel = lin_smooth_kernel
    result = linearized_refine_batch(x_anchors, y_obs, loc, setting, args, device)
    (args.lin_steps, args.lin_particles, args.lin_scale,
     args.lin_sigma, args.lin_alpha, args.lin_gamma,
     args.lin_trust_rms, args.lin_smooth_kernel) = orig
    return result


def proposal_cascade_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """
    Cascade-FlowMap: NO-seeded proposals + multi-scale linearized refinement.

    Step 1: Build mixed ensemble
      • (1 - no_seeded_frac) fraction: pure N(0,I) samples decoded by FlowMap
      • no_seeded_frac fraction: noise NO output at t=no_seed_t, re-flow
        z_seed = (1-t)*x_NO + t*eps  →  x_seed = flow(z_seed, t→0)
        These stay on the FlowMap manifold but start near the NO basin.

    Step 2: Select top-k anchors by coarse projection score (scale=proposal_scale)
      + optional proposal_rounds CEM re-injection

    Step 3: Cascaded multi-scale linearized refinement (frequency continuation)
      for scale in cascade_scales:           # e.g. [16, 8, 4, 1]
          x = linearized_refine(x, scale=scale, steps=cascade_steps, p=cascade_particles)
      Each step: ensemble Gauss-Newton iteration with randomized Jacobian
          dx = α * (D_c D_c^T + (p-1)γI)^{-1} (y-H(x)) D_c^T X_c

    Step 4: Optional clean variational refine then select best by misfit
    """
    n_total   = max(1, args.ensemble)
    n_seed    = int(n_total * getattr(args, 'no_seeded_frac', 0.0))
    n_rand    = n_total - n_seed
    t_seed    = float(getattr(args, 'no_seed_t', 0.4))

    # ── Phase 1: draw random prior proposals ─────────────────────────────────
    z_rand   = torch.randn(n_rand, *x_bg.shape[1:], device=device)
    x_rand   = flow(flow_model, z_rand, 1.0, 0.0).clamp(-1, 1)

    # ── Phase 2: NO-seeded proposals (if available) ──────────────────────────
    if n_seed > 0 and x_bg.abs().mean() > 1e-4:
        x_no_exp = x_bg.expand(n_seed, -1, -1, -1)
        eps      = torch.randn_like(x_no_exp)
        z_seed   = ((1.0 - t_seed) * x_no_exp + t_seed * eps).clamp(-1.5, 1.5)
        # Also add small jitter so seeds don't collapse
        if getattr(args, 'seed_jitter', 0.0) > 0:
            z_seed = z_seed + args.seed_jitter * torch.randn_like(z_seed)
        x_seeded = flow(flow_model, z_seed, t_seed, 0.0).clamp(-1, 1)
        x_all    = torch.cat([x_rand, x_seeded], 0)
    else:
        x_all = x_rand

    # ── Phase 3: CEM re-injection rounds ─────────────────────────────────────
    k = max(1, min(args.proposal_topk, x_all.shape[0]))
    for _ in range(args.proposal_rounds):
        elite_idx = proposal_union_indices(x_all, y_obs, loc, setting, args, device, k)
        x_elite   = x_all[elite_idx]
        draw      = torch.randint(0, x_elite.shape[0], (n_rand,), device=device)
        x_seed2   = x_elite[draw]
        eps2      = torch.randn_like(x_seed2)
        t_ri      = float(args.proposal_reinject_t)
        z_ri      = (1.0 - t_ri) * x_seed2 + t_ri * eps2
        x_new     = flow(flow_model, z_ri, t_ri, 0.0).clamp(-1, 1)
        x_all     = torch.cat([x_all, x_new], 0)

    # ── Phase 4: select top-k anchors ────────────────────────────────────────
    k       = max(1, min(args.proposal_topk, x_all.shape[0]))
    idx     = proposal_union_indices(x_all, y_obs, loc, setting, args, device, k)
    x_elite = x_all[idx]

    # ── Phase 5: cascaded multi-scale linearized refinement ──────────────────
    # Save pre-cascade anchors as safety fallback: if cascade pushes models into
    # wrong local minima (coarse-scale improvement ≠ fine-scale improvement),
    # we revert to the original proposals.
    x_elite_pre = x_elite.clone()

    cascade_scales  = getattr(args, 'cascade_scales',  [args.lin_scale])
    cascade_steps   = getattr(args, 'cascade_steps',   args.lin_steps)
    cascade_parts   = getattr(args, 'cascade_particles', args.lin_particles)
    cascade_sigma   = getattr(args, 'cascade_sigma',   args.lin_sigma)
    cascade_alpha   = getattr(args, 'cascade_alpha',   args.lin_alpha)
    cascade_gamma   = getattr(args, 'cascade_gamma',   args.lin_gamma)
    cascade_trust   = getattr(args, 'cascade_trust',   args.lin_trust_rms)
    cascade_kernel  = getattr(args, 'cascade_kernel',  args.lin_smooth_kernel)

    # Evaluate full-resolution misfit of pre-cascade anchors once (for per-level gate)
    cascade_per_level_gate = getattr(args, 'cascade_per_level_gate', False)
    if cascade_per_level_gate:
        pred_pre0 = predict_obs(x_elite, loc, setting, args, device)
        mis_cur = per_sample_misfit(pred_pre0, y_obs, args)  # shape [k]
    else:
        mis_cur = None

    for lvl, scale in enumerate(cascade_scales):
        # Use slightly smaller alpha at finer scales (more conservative near convergence)
        alpha_lvl = cascade_alpha * (0.8 ** lvl)
        x_candidate = linearized_refine_at_scale(
            x_elite, y_obs, loc, setting,
            scale, cascade_steps, cascade_parts,
            cascade_sigma, alpha_lvl, cascade_gamma, cascade_trust, cascade_kernel,
            args.update_gate, args.update_gate_tol, args, device,
        )
        if cascade_per_level_gate:
            # Per-level, per-model full-resolution gate: only accept cascade update
            # when the candidate strictly improves full-resolution misfit for that model.
            pred_cand = predict_obs(x_candidate, loc, setting, args, device)
            mis_cand = per_sample_misfit(pred_cand, y_obs, args)  # shape [k]
            improve = mis_cand <= mis_cur * (1.0 + float(getattr(args, 'update_gate_tol', 0.0)))
            gate = improve.view(-1, 1, 1, 1)
            x_elite = torch.where(gate, x_candidate, x_elite)
            mis_cur = torch.where(improve, mis_cand, mis_cur)
        else:
            x_elite = x_candidate

    # Safety net: pool pre-cascade and post-cascade candidates, select best k
    # by FULL-RESOLUTION misfit.  This guarantees the cascade can only help.
    x_pool = torch.cat([x_elite_pre, x_elite], dim=0)
    pred_pool = predict_obs(x_pool, loc, setting, args, device)
    mis_pool = per_sample_misfit(pred_pool, y_obs, args)
    best_k = max(1, min(k, x_pool.shape[0]))
    best_idx = torch.topk(-mis_pool, k=best_k).indices
    x_elite = x_pool[best_idx]

    # ── Phase 6: optional clean variational refinement ───────────────────────
    x_ref = clean_refine_batch(x_elite, y_obs, loc, setting, args, device)
    return select_by_misfit(x_ref, y_obs, loc, setting, args, device)


def proposal_smc_case(flow_model, x_bg, y_obs, loc, setting, args, device):
    """
    Sequential Monte Carlo (SMC) posterior sampling via frequency continuation.

    Theory: We approximate the posterior p(x|y) ∝ p(y|x) p(x) via SMC.
    1. Sample N particles from the prior p(x) using the FlowMap model.
    2. Sequentially temper the likelihood across coarse-to-fine scales:
         w_i^(s) = exp(-misfit_s(x_i) / τ_s)
       Resample particles by these weights (high-weight particles survive).
       Optionally apply 1-2 steps of local GN refinement after each resampling.
    3. Return the particle with lowest full-resolution data misfit.

    Advantages over cascade GN:
    - Never destroys good particles (resampling is a soft selection, not an update)
    - With N=1024 prior particles and correct resampling, can approximate posterior
    - The prior (FlowMap) guarantees particles stay on the learned manifold
    - Per-level resampling avoids cycle-skipping by progressive scale refinement
    """
    N = max(1, args.ensemble)
    k = max(1, min(getattr(args, 'smc_elite_k', args.proposal_topk), N))
    tau = float(getattr(args, 'smc_temperature', 0.05))
    smc_scales = getattr(args, 'smc_scales', [16, 8, 4])
    smc_refine_steps = int(getattr(args, 'smc_refine_steps', 1))
    smc_jitter = float(getattr(args, 'smc_particle_jitter', getattr(args, 'smc_jitter', 0.003)))

    # ── Phase 1: sample N particles from prior ───────────────────────────────
    z = torch.randn(N, *x_bg.shape[1:], device=device)
    x_particles = flow(flow_model, z, 1.0, 0.0).clamp(-1, 1)

    # ── Phase 2: SMC across scales ────────────────────────────────────────────
    for scale in smc_scales:
        # Score all particles at current scale
        pred = predict_obs(x_particles, loc, setting, args, device)
        pred_s = prepare_obs_for_assim(pred, scale, args)
        obs_s = prepare_obs_for_assim(y_obs, scale, args)
        log_w = -normed_misfit(pred_s, obs_s, args) / max(tau, 1e-8)
        w = torch.softmax(log_w, dim=0)  # normalized weights

        # Stratified resampling: draw N indices proportional to weights
        idx = torch.multinomial(w, N, replacement=True)
        x_particles = x_particles[idx]

        # Optional local GN refinement (1 step) at this scale to move toward data
        if smc_refine_steps > 0:
            # Temporarily set scale params and run one linearized step
            orig_steps = args.lin_steps
            args.lin_steps = smc_refine_steps
            x_particles = linearized_refine_at_scale(
                x_particles, y_obs, loc, setting,
                scale, smc_refine_steps, args.lin_particles,
                args.lin_sigma, args.lin_alpha * 0.5,  # conservative alpha
                args.lin_gamma, args.lin_trust_rms, args.lin_smooth_kernel,
                args.update_gate, args.update_gate_tol, args, device,
            )
            args.lin_steps = orig_steps

        # Add small jitter to maintain particle diversity (avoid collapse)
        if smc_jitter > 0:
            jitter = smooth_noise_like(x_particles, args.lin_smooth_kernel) * smc_jitter
            x_particles = (x_particles + jitter).clamp(-1, 1)

    # ── Phase 3: final selection – pick best k by full-resolution misfit ──────
    pred_final = predict_obs(x_particles, loc, setting, args, device)
    mis_final = per_sample_misfit(pred_final, y_obs, args)
    top_idx = torch.topk(-mis_final, k=k).indices
    x_elite = x_particles[top_idx]

    x_ref = clean_refine_batch(x_elite, y_obs, loc, setting, args, device)
    return select_by_misfit(x_ref, y_obs, loc, setting, args, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["f", "loc"], default="loc")
    p.add_argument("--flow_ckpt", default="/workspace/fmm_outputs/bench_fvb_flowmap_fast/fmtt/final.pt")
    p.add_argument("--no_ckpt", default="/workspace/fmm_outputs/bench_fvb_operator/unet/final.pt")
    p.add_argument("--cache_dir", default="/data10/fwi_cache/fvb")
    p.add_argument("--data_root", default="/workspace/fdo-fwi/data/fvb")
    p.add_argument("--out", default="/workspace/fmm_outputs/fmda_shift.json")
    p.add_argument("--save_pred_dir", default="",
                   help="Optional directory for per-case gt/no/fmda_auto prediction .npy files.")
    p.add_argument("--skip", type=int, default=10000)
    p.add_argument("--n_test", type=int, default=16)
    p.add_argument("--method", choices=["eki", "var", "smc", "mode", "proposal", "localmap", "cascade", "psmc", "proposal_ms", "proposal_ms_adam", "proposal_ms_eki", "proposal_ms_lgfmi", "proposal_ms_lgfmi_grad", "proposal_ms_fdps", "proposal_ms_svgd"], default="eki")
    p.add_argument("--bg_mode", choices=["no", "smooth", "auto", "zero"], default="no")
    p.add_argument("--only_auto", action="store_true")
    p.add_argument("--ensemble", type=int, default=8)
    p.add_argument("--ensemble_spread", type=float, default=0.02)
    p.add_argument("--ode_steps", type=int, default=16)
    p.add_argument("--t_start", type=float, default=0.6)
    p.add_argument("--assim_start", type=float, default=1.0)
    p.add_argument("--times", default="0.6,0.3,0.0")
    p.add_argument("--obs_scales", default="1,1")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--update_gate", action="store_true")
    p.add_argument("--update_gate_tol", type=float, default=0.0)
    p.add_argument("--select", choices=["mean", "best", "soft", "random"], default="mean")
    p.add_argument("--select_temp", type=float, default=0.01)
    p.add_argument("--robust_select", action="store_true",
                   help="Final blind selection by rank aggregation over focused/global/relative multiscale views.")
    p.add_argument("--eki_alpha", type=float, default=0.7)
    p.add_argument("--eki_time_power", type=float, default=0.0)
    p.add_argument("--eki_gamma", type=float, default=0.05)
    p.add_argument("--trust_rms", type=float, default=0.0)
    p.add_argument("--trust_ratio", type=float, default=0.0)
    p.add_argument("--eki_inner", type=int, default=1)
    p.add_argument("--eki_mda", action="store_true")
    p.add_argument("--eki_inflate", type=float, default=1.0)
    p.add_argument("--obs_whiten", action="store_true")
    p.add_argument("--smc_temp", type=float, default=0.01)
    p.add_argument("--smc_jitter", type=float, default=0.01)
    p.add_argument("--smc_ess", type=float, default=0.75)
    p.add_argument("--smc_repeats", type=int, default=1)
    p.add_argument("--mode_select", choices=["topk", "soft"], default="topk")
    p.add_argument("--mode_topk", type=int, default=8)
    p.add_argument("--mode_temp", type=float, default=0.01)
    p.add_argument("--mode_jitter", type=float, default=0.0)
    p.add_argument("--mode_repeats", type=int, default=1)
    p.add_argument("--mode_noise", choices=["fresh", "selected", "mix"], default="fresh")
    p.add_argument("--mode_noise_mix", type=float, default=0.25)
    p.add_argument("--mode_blend", type=float, default=1.0)
    p.add_argument("--var_opt", choices=["adam", "lbfgs", "pgd_smooth"], default="adam")
    p.add_argument("--var_steps", type=int, default=6)
    p.add_argument("--var_lr", type=float, default=0.02)
    p.add_argument("--var_gamma", type=float, default=0.05)
    p.add_argument("--var_bg", type=float, default=0.10)
    p.add_argument("--var_grad_clip", type=float, default=1.0)
    p.add_argument("--var_smooth_kernel", type=int, default=5)
    p.add_argument("--localmap_steps", type=int, default=2)
    p.add_argument("--localmap_lr", type=float, default=0.005)
    p.add_argument("--localmap_gamma", type=float, default=1.0)
    p.add_argument("--localmap_bg", type=float, default=0.5)
    p.add_argument("--localmap_grad_clip", type=float, default=0.5)
    p.add_argument("--localmap_smooth_kernel", type=int, default=5)
    p.add_argument("--proposal_scale", type=int, default=16)
    p.add_argument("--proposal_score_scales", default="")
    p.add_argument("--proposal_union", action="store_true")
    p.add_argument("--proposal_pool_fallback", action="store_true",
                   help="Pool pre- and post-GN elite candidates; select best at full resolution. "
                        "Prevents GN divergence from corrupting easy-case good prior samples.")
    p.add_argument("--proposal_ms_scales", default="16,8,4,1",
                   help="Comma-separated scales for multi-scale proposal method (proposal_ms)")
    p.add_argument("--proposal_ms_topk_start", type=int, default=256,
                   help="Starting topk for multi-scale proposal (coarsest level)")
    p.add_argument("--proposal_ms_union_k", type=int, default=0,
                   help="If >0, initial pool = union of top-k from coarsest scale AND top-k from "
                        "union_scale. Robustly handles both hard (coarse reliable) and easy "
                        "(coarse may cycle-skip) cases. 0 = sequential narrowing (default).")
    p.add_argument("--proposal_ms_union_scale", type=int, default=4,
                   help="Second scale for union initial selection (default 4). Used only when "
                        "proposal_ms_union_k > 0 and proposal_ms_union_views not set.")
    p.add_argument("--proposal_ms_union_views", default=None,
                   help="Comma-separated list of scales for N-way union initial selection. "
                        "E.g. '16,8,4,2' → 4 views each contributing union_k/4 candidates. "
                        "Overrides proposal_ms_union_scale when set.")
    p.add_argument("--proposal_topk", type=int, default=8)
    p.add_argument("--proposal_rounds", type=int, default=0)
    p.add_argument("--proposal_reinject_t", type=float, default=0.4)
    p.add_argument("--proposal_jitter", type=float, default=0.0)
    p.add_argument("--lin_steps", type=int, default=0)
    p.add_argument("--lin_particles", type=int, default=8)
    p.add_argument("--lin_sigma", type=float, default=0.03)
    p.add_argument("--lin_alpha", type=float, default=0.5)
    p.add_argument("--lin_gamma", type=float, default=1e-3)
    p.add_argument("--lin_scale", type=int, default=4)
    p.add_argument("--lin_trust_rms", type=float, default=0.03)
    p.add_argument("--lin_smooth_kernel", type=int, default=7)
    p.add_argument("--clean_steps", type=int, default=0)
    p.add_argument("--clean_lr", type=float, default=0.01)
    p.add_argument("--clean_bg", type=float, default=0.1)
    p.add_argument("--clean_scale", type=int, default=1)
    p.add_argument("--select_score_scales", default="")
    p.add_argument("--clean_grad_clip", type=float, default=1.0)
    p.add_argument("--lin_full_gate", action="store_true",
                   help="Full-resolution (scale=1) acceptance gate for GN steps. "
                        "Required when using large trust_rms to prevent cycle-skip acceptance. "
                        "Costs 1 extra FWM per model per step but allows trust_rms up to 0.1+.")
    # ── Adam refinement args (proposal_ms_adam) ────────────────────────────────
    p.add_argument("--adam_steps", type=int, default=8,
                   help="Adam refinement steps per candidate (default 8). "
                        "Cost: adam_k * adam_steps FWM forward+backward passes.")
    p.add_argument("--adam_lr", type=float, default=0.005,
                   help="Adam learning rate (default 0.005). Scales naturally: "
                        "easy cases have small gradient → small step even with larger lr.")
    p.add_argument("--adam_bg", type=float, default=0.5,
                   help="L2 regularization weight toward initial position (default 0.5). "
                        "Prevents drift for easy cases; reduce for hard cases.")
    p.add_argument("--adam_scale", type=int, default=4,
                   help="Spatial scale for Adam misfit (default 4). Coarser=faster, less cycle-skip.")
    p.add_argument("--adam_scale2", type=int, default=0,
                   help="Second scale for dual-scale Adam loss (0=disabled). E.g. 1 for full-res.")
    p.add_argument("--adam_k", type=int, default=64,
                   help="Number of elite candidates for Adam refinement (default 64).")
    p.add_argument("--adam_gate", action="store_true", default=True,
                   help="Full-res gate for Adam: revert if misfit worsens (default True).")
    # ── EKI refinement args (proposal_ms_eki) ─────────────────────────────────
    p.add_argument("--eki_steps", type=int, default=3,
                   help="ES-MDA steps for EKI refinement (default 3). "
                        "Step 1 is FREE (reuses pred_all). Steps 2+ cost eki_k FWM each.")
    p.add_argument("--eki_sigma", type=float, default=0.1,
                   help="Base observation noise for EKI/ES-MDA (default 0.1). "
                        "Effective noise per step = sigma * sqrt(eki_steps).")
    p.add_argument("--eki_k", type=int, default=128,
                   help="EKI ensemble size — candidates kept before EKI update (default 128).")
    p.add_argument("--eki_adaptive_sigma", action="store_true", default=False,
                   help="Scale EKI sigma adaptively: larger sigma for easy cases (weak signal) "
                        "to limit Kalman gain and prevent overshoot. "
                        "sigma_adapt = base * max(1, ref_rms / innov_rms).")
    p.add_argument("--eki_adaptive_ref_rms", type=float, default=0.15,
                   help="Reference innovation RMS for adaptive sigma (default 0.15 = typical mid case).")
    p.add_argument("--eki_adaptive_sigma_max", type=float, default=4.0,
                   help="Maximum multiplier for adaptive sigma (default 4x base sigma).")
    p.add_argument("--post_eki_adam_steps", type=int, default=0,
                   help="Post-EKI Adam polish steps on best candidate (default 0=off). "
                        "Very cheap: 1 candidate × steps gradient FWM calls. "
                        "Adam auto-scales: tiny gradient for easy cases → no overshoot.")
    p.add_argument("--post_eki_adam_lr", type=float, default=0.003,
                   help="Adam lr for post-EKI polish (default 0.003).")
    p.add_argument("--post_eki_adam_bg", type=float, default=0.2,
                   help="Background regularization for post-EKI polish (default 0.2).")
    # ── LG-FMI args (proposal_ms_lgfmi) ───────────────────────────────────────
    p.add_argument("--lgfmi_t_mid", type=float, default=0.5,
                   help="Intermediate flow time for LG-FMI EKI update (default 0.5). "
                        "EKI applied at x_t_mid instead of x_0 → stays on ODE manifold.")
    p.add_argument("--lgfmi_k", type=int, default=128,
                   help="Number of candidates for LG-FMI intermediate-time EKI (default 128). "
                        "Uses eki_k if not set.")
    p.add_argument("--lgfmi_k_inner", type=int, default=0,
                   help="Second filter size after endpoint-preview scoring (0=disabled). "
                        "If >0, score all lgfmi_k by H(x̂_1) and keep only best lgfmi_k_inner.")
    p.add_argument("--lgfmi_sigma", type=float, default=0.2,
                   help="EKI observation noise for LG-FMI update (default 0.2). "
                        "Same role as eki_sigma but applied in x_t space.")
    p.add_argument("--lgfmi_steps", type=int, default=1,
                   help="EKI steps at the intermediate time stage (default 1). "
                        "Each step after the first costs lgfmi_k velocity+FWM calls.")
    p.add_argument("--lgfmi_scale", type=int, default=4,
                   help="Observation scale for LG-FMI EKI update (default 4 = lin_scale).")
    p.add_argument("--lgfmi_v_thresh", type=float, default=0.0,
                   help="Innovation-RMS threshold for leverage gate (0=disabled). "
                        "Skip EKI update when innovation RMS < threshold (already-good fit).")
    p.add_argument("--lgfmi_gate", action="store_true", default=True,
                   help="Gate: pool pre- and post-LG-FMI candidates, select best (default True).")
    # ── LG-FMI-Grad args (proposal_ms_lgfmi_grad) ─────────────────────────────
    p.add_argument("--lgfmi_grad_steps", type=int, default=5,
                   help="Adam steps per candidate in x_t space (default 5). "
                        "Cost: lgfmi_k × lgfmi_grad_steps × ~3× FWM (fwd+bwd).")
    p.add_argument("--lgfmi_grad_lr", type=float, default=0.01,
                   help="Adam learning rate for x_t gradient update (default 0.01).")
    p.add_argument("--lgfmi_grad_trust", type=float, default=0.2,
                   help="Trust-region RMS for x_t update (default 0.2). "
                        "||x_t_new - x_t_init||_rms ≤ trust_rms after each step.")
    p.add_argument("--lgfmi_grad_bg", type=float, default=0.05,
                   help="L2 regularization toward x_t_init (default 0.05). "
                        "Prevents large manifold deviations; easy cases auto-restrain "
                        "via small gradient magnitude.")
    # FlowMap-DPS station guidance args.  This is the clean DPS analogue:
    # x_t -> endpoint preview -> H -> loss -> grad wrt x_t, at a few stations.
    p.add_argument("--fdps_k", type=int, default=24)
    p.add_argument("--fdps_times", default="0.4,0.2,0.1")
    p.add_argument("--fdps_scales", default="16,8,4")
    p.add_argument("--fdps_steps_per_t", type=int, default=3)
    p.add_argument("--fdps_lr", default="0.02",
                   help="Scalar or comma schedule per station.")
    p.add_argument("--fdps_trust", default="0.05",
                   help="Scalar or comma RMS trust schedule per station.")
    p.add_argument("--fdps_bg", default="0.0",
                   help="Scalar or comma L2 prior schedule per station.")
    p.add_argument("--fdps_opt", choices=["adam", "sgd", "momentum"], default="adam")
    p.add_argument("--fdps_reselect", action="store_true")
    p.add_argument("--fdps_retain", type=int, default=24)
    p.add_argument("--fdps_reinject", type=float, default=0.0)
    p.add_argument("--fdps_repulse", type=float, default=0.0)
    p.add_argument("--fdps_gate", action="store_true",
                   help="Keep a station update only when its endpoint score improves.")
    p.add_argument("--fdps_gate_tol", type=float, default=0.0)
    p.add_argument("--fdps_gate_scales", default="",
                   help="Comma score scales for FDPS station gate; default=current scale.")
    p.add_argument("--fdps_post_adam_steps", type=int, default=0,
                   help="Optional short clean-space Adam polish on best FDPS endpoints.")
    p.add_argument("--fdps_post_adam_k", type=int, default=8)
    p.add_argument("--fdps_post_adam_lr", type=float, default=0.0015)
    p.add_argument("--fdps_post_adam_bg", type=float, default=0.35)
    p.add_argument("--fdps_post_adam_scale", type=int, default=4)
    p.add_argument("--fdps_post_adam_scale2", type=int, default=0)
    # C2F-SVGD-FM: differentiable H + multi-time z_t updates + SVGD particles.
    p.add_argument("--svgd_k", type=int, default=16,
                   help="Number of particles kept for C2F-SVGD-FM.")
    p.add_argument("--svgd_times", default="0.85,0.70,0.55,0.40,0.25,0.12",
                   help="Descending flow-time stations for C2F-SVGD-FM.")
    p.add_argument("--svgd_scales", default="8,6,4,3,2,1",
                   help="Coarse-to-fine observation scales for C2F-SVGD-FM.")
    p.add_argument("--svgd_steps", default="4,4,3,3,2,2",
                   help="SVGD inner steps per flow-time station.")
    p.add_argument("--svgd_lr", default="0.03,0.025,0.020,0.015,0.010,0.006",
                   help="Normalized SVGD step size per station.")
    p.add_argument("--svgd_prior", default="0.05,0.08,0.12,0.20,0.35,0.50",
                   help="Trajectory tether weight per station.")
    p.add_argument("--svgd_trust", default="0.035,0.030,0.025,0.020,0.015,0.012",
                   help="RMS trust radius around each station's entry state.")
    p.add_argument("--svgd_feature", type=int, default=12,
                   help="Pooled feature size for the SVGD RBF kernel.")
    p.add_argument("--svgd_bw", type=float, default=1.0,
                   help="Bandwidth multiplier for median-heuristic RBF kernel.")
    p.add_argument("--svgd_repulse", type=float, default=1.0,
                   help="Weight of SVGD kernel repulsion term.")
    p.add_argument("--svgd_grad_clip", type=float, default=0.0,
                   help="Optional RMS clip for score gradient before SVGD.")
    p.add_argument("--svgd_gate", action="store_true", default=True,
                   help="Accept/reject each station by blind projected FWM score.")
    p.add_argument("--svgd_gate_tol", type=float, default=0.0)
    p.add_argument("--svgd_gate_scales", default="",
                   help="Optional multiscale gate scales; default uses current scale.")
    # DPS-style enhancements for _c2f_svgd_refine_xt
    p.add_argument("--svgd_tmpd_rho", type=float, default=0.0,
                   help="TMPD: x0 += rho*(y-H(x0)) residual correction after look-ahead.")
    p.add_argument("--svgd_ps_k", type=int, default=1,
                   help="PS+: average gradient over K noisy look-aheads (1=standard).")
    p.add_argument("--svgd_ps_sigma", type=float, default=0.01,
                   help="PS+ noise sigma for K look-ahead averaging.")
    p.add_argument("--svgd_mcg_steps", type=int, default=0,
                   help="MCG: manifold-projection steps after SVGD update (0=off).")
    p.add_argument("--svgd_mcg_lr", type=float, default=0.02,
                   help="MCG manifold-projection step size.")
    p.add_argument("--svgd_optim", default="svgd",
                   choices=["svgd", "adam", "eki", "pgd"],
                   help="Optimizer in z_t space: svgd/adam/eki/pgd.")
    p.add_argument("--svgd_adam_b1", type=float, default=0.9,
                   help="Adam beta1 (momentum decay).")
    p.add_argument("--svgd_adam_b2", type=float, default=0.999,
                   help="Adam beta2 (second-moment decay).")
    p.add_argument("--svgd_eki_gamma", type=float, default=0.1,
                   help="EKI observation noise level (regularization).")
    p.add_argument("--svgd_eki_scale", type=int, default=4,
                   help="EKI coarse scale for H evaluation (cheaper).")
    # Advanced C2F-SVGD options
    p.add_argument("--svgd_grad_smooth", action="store_true", default=False,
                   help="Low-pass filter gradient at large t (physics: large t = low freq only).")
    p.add_argument("--svgd_hybrid_rep", type=float, default=0.0,
                   help="Add SVGD repulsion ON TOP of Adam/PGD direction (hybrid).")
    p.add_argument("--svgd_adaptive_lr", action="store_true", default=False,
                   help="Scale lr by gradient rms (preserve gradient magnitude info).")
    p.add_argument("--svgd_resample", action="store_true", default=False,
                   help="SMC-style resample particles after each station gate.")
    p.add_argument("--svgd_x0_kernel", action="store_true", default=False,
                   help="Compute SVGD kernel in x0 (velocity) space instead of z_t space.")
    p.add_argument("--svgd_carry_momentum", action="store_true", default=False,
                   help="Carry Adam momentum state across time stations.")
    p.add_argument("--svgd_grad_clip_rms", type=float, default=0.0,
                   help="Clip gradient RMS (don't normalize; preserve scale). 0=normalize.")
    p.add_argument("--svgd_x0_adam_steps", type=int, default=0,
                   help="Post-SVGD: Adam steps directly in x0 (velocity) space. 0=off.")
    p.add_argument("--svgd_x0_adam_lr", type=float, default=0.005,
                   help="Learning rate for x0-space Adam post-processing.")
    p.add_argument("--svgd_x0_adam_scale", type=int, default=4,
                   help="Observation scale for x0-space Adam (coarser=faster).")
    p.add_argument("--svgd_lowfreq_t_thresh", type=float, default=0.3,
                   help="Apply low-freq gradient filter only at t > this threshold.")
    # Noisy Adam (Langevin + Adam)
    p.add_argument("--svgd_noise_temp", type=float, default=0.0,
                   help="Langevin noise temperature at t=0.85 (scales as t/0.85). 0=off.")
    # Cross-particle shared Adam
    p.add_argument("--svgd_shared_adam", action="store_true", default=False,
                   help="Share Adam v (scaling) across particles for better estimate.")
    # SIR resampling
    p.add_argument("--svgd_sir", action="store_true", default=False,
                   help="SIR: resample particles by gate score after each station.")
    p.add_argument("--svgd_sir_jitter", type=float, default=0.005,
                   help="Jitter sigma after SIR resampling.")
    # Trajectory-aware momentum
    p.add_argument("--svgd_carry_mom", action="store_true", default=False,
                   help="Carry Adam momentum across time stations via flow propagation.")
    # Bidirectional FlowMap re-noising for basin escape
    p.add_argument("--svgd_bidir_rounds", type=int, default=0,
                   help="Bidirectional rounds: blind-select clean elites, re-noise to t, then rerun C2F tail.")
    p.add_argument("--svgd_bidir_times", default="0.70",
                   help="Comma-separated re-noising times for bidirectional C2F rounds.")
    p.add_argument("--svgd_bidir_k", type=int, default=16,
                   help="Number of clean elites used for each bidirectional re-noising round.")
    p.add_argument("--svgd_bidir_jitter", type=float, default=0.015,
                   help="Jitter sigma applied at the re-noised FlowMap state.")
    p.add_argument("--svgd_bidir_temp", type=float, default=0.02,
                   help="Temperature for duplicating bidirectional elites when needed.")
    p.add_argument("--svgd_bidir_select_scales", default="",
                   help="Multiscale blind-select scales for bidirectional elites; default uses select_score_scales.")
    p.add_argument("--svgd_bidir_carry_momentum", action="store_true", default=False,
                   help="Carry Adam state into bidirectional reruns instead of resetting at re-entry.")
    # ── Cascade method args ────────────────────────────────────────────────────
    p.add_argument("--no_seeded_frac",  type=float, default=0.0,
                   help="Fraction of ensemble seeded from NO output (0=pure random)")
    p.add_argument("--no_seed_t",       type=float, default=0.4,
                   help="Noise level t for NO-seeded proposals (higher=more noise)")
    p.add_argument("--seed_jitter",     type=float, default=0.0,
                   help="Extra jitter for NO seeds to avoid collapse")
    p.add_argument("--cascade_scales",  default="",
                   help="Comma-separated observation downscale factors for cascade (e.g. '16,4,1')")
    p.add_argument("--cascade_steps",   type=int, default=2,
                   help="Linearized steps per cascade level")
    p.add_argument("--cascade_particles", type=int, default=16,
                   help="Perturbation particles per cascade level")
    p.add_argument("--cascade_sigma",   type=float, default=0.025,
                   help="Perturbation sigma for cascade linearization")
    p.add_argument("--cascade_alpha",   type=float, default=0.45,
                   help="Step size alpha for cascade (decays 0.8x per finer level)")
    p.add_argument("--cascade_gamma",   type=float, default=1e-3,
                   help="Regularization gamma for cascade Gauss-Newton")
    p.add_argument("--cascade_trust",   type=float, default=0.025,
                   help="Trust region RMS for cascade updates")
    p.add_argument("--cascade_kernel",  type=int, default=7,
                   help="Smooth kernel size for cascade perturbations")
    p.add_argument("--cascade_per_level_gate", action="store_true",
                   help="Per-level full-resolution gate: revert per-model if fine-scale misfit worsens")
    # SMC args
    p.add_argument("--smc_temperature", type=float, default=0.05,
                   help="Temperature tau for SMC likelihood weighting: w ∝ exp(-misfit/tau)")
    p.add_argument("--smc_scales", default="16,8,4",
                   help="Comma-separated scales for SMC sequential tempering")
    p.add_argument("--smc_elite_k", type=int, default=16,
                   help="Number of elite particles for final selection in SMC")
    p.add_argument("--smc_refine_steps", type=int, default=1,
                   help="GN refinement steps per SMC level (0=none)")
    p.add_argument("--smc_particle_jitter", type=float, default=0.003,
                   help="Jitter std added after each SMC resampling for diversity")
    p.add_argument("--nt", type=int, default=300)
    p.add_argument("--sampling_rate", type=int, default=2)
    p.add_argument("--freq", type=float, default=15.0)
    p.add_argument("--misfit_mode", choices=["global", "focused", "relative"], default="global")
    p.add_argument("--score_domain", choices=["norm", "rawstd"], default="norm")
    p.add_argument("--psd_score", choices=["none", "rank"], default="none",
                   help="Use PSD/ESD band residual as a blind rank/gate score. "
                        "Default none preserves previous behavior.")
    p.add_argument("--psd_mix", type=float, default=0.5,
                   help="Mixture weight for PSD score when --psd_score rank.")
    p.add_argument("--psd_bands", default="0.0-0.18,0.18-0.45,0.45-1.0",
                   help="Comma frequency bands as low-high fractions of Nyquist.")
    p.add_argument("--psd_weights", default="1.0,0.7,0.25",
                   help="Weights for --psd_bands.")
    p.add_argument("--psd_split_rank", action="store_true",
                   help="During proposal union, also keep top candidates from each PSD band.")
    p.add_argument("--psd_split_scale", type=int, default=16,
                   help="Projection scale used by --psd_split_rank.")
    p.add_argument("--direct_mute_frac", type=float, default=0.0)
    p.add_argument("--late_weight", type=float, default=0.0)
    args = p.parse_args()
    args.times = [float(x) for x in args.times.split(",")]
    args.obs_scales = [int(x) for x in args.obs_scales.split(",")]
    args.proposal_score_scales = parse_int_list(args.proposal_score_scales)
    args.select_score_scales   = parse_int_list(args.select_score_scales)
    args.cascade_scales        = parse_int_list(args.cascade_scales) if args.cascade_scales else []
    args.smc_scales            = parse_int_list(args.smc_scales) if args.smc_scales else [16, 8, 4]
    args.proposal_ms_scales    = parse_int_list(args.proposal_ms_scales) if args.proposal_ms_scales else [16, 8, 4, 1]
    args.proposal_ms_union_views = parse_int_list(args.proposal_ms_union_views) if args.proposal_ms_union_views else None
    args.nbc = 120
    args.dx = 10
    args.dt = 1e-3
    args.sz = 10
    args.gz = 10
    grids = 70
    args.sx = np.linspace(0, grids - 1, 5) * args.dx
    args.gx = np.linspace(0, grids - 1, grids) * args.dx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_all, y_f_all = load_block(args.cache_dir, args.skip, args.n_test)
    loc_all = load_loc(args.data_root, args.skip, args.n_test)

    ckpt = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    flow_model = TinyUNetVel(base=int(ckpt_args.get("base", 64)), emb_dim=int(ckpt_args.get("emb_dim", 192))).to(device).eval()
    flow_model.ode_steps = args.ode_steps
    flow_model.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
    for param in flow_model.parameters():
        param.requires_grad_(False)

    no_model = UNet().to(device).eval()
    no_model.load_state_dict(torch.load(args.no_ckpt, map_location=device, weights_only=False)["model"])
    for param in no_model.parameters():
        param.requires_grad_(False)

    rows = []
    for i in range(args.n_test):
        x = x_all[i : i + 1].to(device)
        loc = loc_all[i : i + 1]
        y_obs = make_observation(x, y_f_all[i : i + 1], loc, args.setting, args, device)
        no_pred = no_model(obs_image(y_obs)).clamp(-1, 1)
        smooth = smooth_bg(no_pred)
        if args.bg_mode == "auto":
            no_bg_mis = per_sample_misfit(predict_obs(no_pred, loc, args.setting, args, device), y_obs, args).mean()
            sm_bg_mis = per_sample_misfit(predict_obs(smooth, loc, args.setting, args, device), y_obs, args).mean()
            bg_auto = smooth.detach() if float(sm_bg_mis.cpu()) < float(no_bg_mis.cpu()) else no_pred.detach()
        elif args.bg_mode == "zero":
            bg_auto = torch.zeros_like(no_pred)
        else:
            bg_auto = no_pred.detach() if args.bg_mode == "no" else smooth.detach()
        if args.only_auto:
            fmda_no, mis_no = no_pred.detach(), float("nan")
            fmda_smooth, mis_sm = smooth.detach(), float("nan")
            if args.method == "eki":
                runner = fmda_case
            elif args.method == "smc":
                runner = fmda_smc_case
            elif args.method == "mode":
                runner = fmda_mode_case
            elif args.method == "proposal":
                runner = proposal_case
            elif args.method == "cascade":
                runner = proposal_cascade_case
            elif args.method == "psmc":
                runner = proposal_smc_case
            elif args.method == "proposal_ms":
                runner = proposal_ms_case
            elif args.method == "proposal_ms_adam":
                runner = proposal_ms_adam_case
            elif args.method == "proposal_ms_eki":
                runner = proposal_ms_eki_case
            elif args.method == "proposal_ms_lgfmi":
                runner = proposal_ms_lgfmi_case
            elif args.method == "proposal_ms_lgfmi_grad":
                runner = proposal_ms_lgfmi_grad_case
            elif args.method == "proposal_ms_fdps":
                runner = proposal_ms_fdps_case
            elif args.method == "proposal_ms_svgd":
                runner = proposal_ms_svgd_case
            elif args.method == "localmap":
                runner = fmda_localmap_case
            else:
                runner = fmda_var_case
            fmda_auto, mis_auto = runner(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "eki":
            fmda_no, mis_no = fmda_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = fmda_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = fmda_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "smc":
            fmda_no, mis_no = fmda_smc_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = fmda_smc_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = fmda_smc_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "mode":
            fmda_no, mis_no = fmda_mode_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = fmda_mode_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = fmda_mode_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal":
            fmda_no, mis_no = proposal_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "cascade":
            fmda_no, mis_no = proposal_cascade_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_cascade_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_cascade_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "psmc":
            fmda_no, mis_no = proposal_smc_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_smc_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_smc_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms":
            fmda_no, mis_no = proposal_ms_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_adam":
            fmda_no, mis_no = proposal_ms_adam_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_adam_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_adam_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_eki":
            fmda_no, mis_no = proposal_ms_eki_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_eki_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_eki_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_lgfmi":
            fmda_no, mis_no = proposal_ms_lgfmi_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_lgfmi_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_lgfmi_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_lgfmi_grad":
            fmda_no, mis_no = proposal_ms_lgfmi_grad_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_lgfmi_grad_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_lgfmi_grad_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_fdps":
            fmda_no, mis_no = proposal_ms_fdps_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_fdps_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_fdps_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "proposal_ms_svgd":
            fmda_no, mis_no = proposal_ms_svgd_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = proposal_ms_svgd_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = proposal_ms_svgd_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        elif args.method == "localmap":
            fmda_no, mis_no = fmda_localmap_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = fmda_localmap_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = fmda_localmap_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        else:
            fmda_no, mis_no = fmda_var_case(flow_model, no_pred.detach(), y_obs, loc, args.setting, args, device)
            fmda_smooth, mis_sm = fmda_var_case(flow_model, smooth.detach(), y_obs, loc, args.setting, args, device)
            fmda_auto, mis_auto = fmda_var_case(flow_model, bg_auto, y_obs, loc, args.setting, args, device)
        row = {
            "i": i,
            "no_mse": float(F.mse_loss(no_pred, x).cpu()),
            "smooth_mse": float(F.mse_loss(smooth, x).cpu()),
            "fmda_no_mse": float(F.mse_loss(fmda_no, x).cpu()),
            "fmda_smooth_mse": float(F.mse_loss(fmda_smooth, x).cpu()),
            "fmda_auto_mse": float(F.mse_loss(fmda_auto, x).cpu()),
            "fmda_no_internal_misfit": mis_no,
            "fmda_smooth_internal_misfit": mis_sm,
            "fmda_auto_internal_misfit": mis_auto,
        }
        if args.save_pred_dir:
            os.makedirs(args.save_pred_dir, exist_ok=True)
            np.save(os.path.join(args.save_pred_dir, f"i{i:02d}_gt.npy"), x.detach().cpu().numpy())
            np.save(os.path.join(args.save_pred_dir, f"i{i:02d}_no.npy"), no_pred.detach().cpu().numpy())
            np.save(os.path.join(args.save_pred_dir, f"i{i:02d}_flowmap.npy"), fmda_auto.detach().cpu().numpy())
            with open(os.path.join(args.save_pred_dir, f"i{i:02d}_row.json"), "w") as f:
                json.dump(row, f, indent=2)
        print(json.dumps(row), flush=True)
        rows.append(row)
    summary = {}
    for k in rows[0]:
        if k == "i":
            continue
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k] = float(vals.mean())
        summary[k + "_std"] = float(vals.std())
    summary["args"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(args).items()}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
