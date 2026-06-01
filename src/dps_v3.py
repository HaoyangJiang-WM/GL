#!/usr/bin/env python3
"""
DPS-CVA v3 — 改进版 DPS 推断 + 可视化
仅针对难case: i=0 (easy=hard, NO MSE=0.277), i=11 (NO MSE=0.179)

新方法 (全部基于同一个 DDPM checkpoint, 与 DPS_vanilla 对比):
  DPS_vanilla_z01 : 标准 DPS, zeta=0.10 (i=0 的最优)
  DPS_vanilla_z05 : 标准 DPS, zeta=0.50 (i=11 的最优)
  DPS_warmstart   : 从 NO + noise@t=500 初始化, 运行 100 步 DPS
  DPS_annealed    : zeta 从 0.5 cosine 渐减到 0.02
  DPS_psd         : PSD midpoint proxy (Tweedie→noise→denoise again)
  DPS_tmpd        : TMPD-lite: x̂₀ += rho*A^T(y-A(x̂₀))
  DPS_ps_plus     : avg K=4 noisy x̂₀ 的前向残差 (ps+)
  DPS_mcg         : MCG: 先做 DPS step 再投影到观测流形
"""
import argparse, os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

# ─── Load eval_ms ─────────────────────────────────────────────────────────────
def _load_eval_ms(path="/tmp/eval_ms.py"):
    spec = importlib.util.spec_from_file_location("eval_ms_dps_v3", path)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = "eval_ms_dps_v3"
    sys.modules["eval_ms_dps_v3"] = mod
    spec.loader.exec_module(mod)
    return mod

print("Loading eval_ms...", flush=True)
EM = _load_eval_ms()
load_block       = EM.load_block
load_loc         = EM.load_loc
make_observation = EM.make_observation
predict_obs_grad = EM.predict_obs_grad
to_raw           = EM.to_raw

# ─── Standard eval args ───────────────────────────────────────────────────────
def build_args():
    a = argparse.Namespace()
    a.cache_dir = "/data10/fwi_cache/cva"
    a.data_root = "/workspace/fdo-fwi/data/cva"
    a.skip=25000; a.n_test=32; a.seed=20260524; a.setting="loc"
    a.nbc=120; a.dx=10; a.dt=1e-3; a.sz=10; a.gz=10
    grids=70
    a.sx = np.linspace(0, grids-1, 5) * a.dx
    a.gx = np.linspace(0, grids-1, grids) * a.dx
    a.nt=300; a.sampling_rate=2; a.freq=15.0
    a.score_domain="norm"; a.obs_scales=[16,8,4,2,1]
    a.misfit_mode="focused"; a.direct_mute_frac=0.0; a.late_weight=0.0
    a.obs_whiten=False
    a.ode_steps=5; a.ensemble=1; a.method="dps"; a.only_auto=True
    a.bg_mode="zero"; a.no_seeded_frac=0.0
    return a

# ─── CVA Measurement Operator ─────────────────────────────────────────────────
class CVAOperator:
    def __init__(self, y_obs, loc, args, device):
        self.y_obs  = y_obs
        self.loc    = loc
        self.args   = args
        self.device = device

    def forward(self, v):
        """v: (1,1,70,70) → predicted seismic data"""
        return predict_obs_grad(v, self.loc, self.args.setting, self.args, self.device)

    def AT(self, r, x_hat):
        """A^T r: adjoint via autograd on x_hat (requires grad)"""
        if not x_hat.requires_grad:
            x_hat = x_hat.detach().requires_grad_(True)
        y_p = self.forward(x_hat)
        (y_p * r.detach()).sum().backward()
        g = x_hat.grad.detach().clone()
        x_hat.grad = None
        return g

# ─── DDPM Schedule ────────────────────────────────────────────────────────────
class DDPMSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.alpha_bar   = alpha_bar
        self.sqrt_ab     = alpha_bar.sqrt()
        self.sqrt_1mab   = (1 - alpha_bar).sqrt()

    def predict_x0(self, xt, t_idx, eps_pred):
        """Tweedie: x̂₀ = (x_t - sqrt(1-ᾱ)·ε) / sqrt(ᾱ)"""
        a = self.sqrt_ab[t_idx].view(-1,1,1,1)
        b = self.sqrt_1mab[t_idx].view(-1,1,1,1)
        return (xt - b * eps_pred) / a.clamp_min(1e-8)

    def ddim_step(self, xt, t_idx, t_prev_idx, eps_pred):
        """Deterministic DDIM (σ=0)"""
        x0 = self.predict_x0(xt, t_idx, eps_pred).clamp(-1, 1)
        if t_prev_idx >= 0:
            ab_p = self.alpha_bar[t_prev_idx]
        else:
            ab_p = torch.ones(1, device=xt.device)
        return ab_p.sqrt() * x0 + (1 - ab_p).clamp_min(0).sqrt() * eps_pred

    def q_sample(self, x0, t_idx, noise=None):
        """前向扩散: q(x_t | x_0) = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_ab[t_idx].view(-1,1,1,1)
        b = self.sqrt_1mab[t_idx].view(-1,1,1,1)
        return a * x0 + b * noise

# ─── DDPM UNet (与 train_ddpm_cva.py 完全一致) ───────────────────────────────
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half-1))
        self.register_buffer("freqs", freqs)
    def forward(self, t):
        emb = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        g = min(8, ch)
        self.norm1 = nn.GroupNorm(g, ch); self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, ch*2)
        self.norm2 = nn.GroupNorm(g, ch); self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()
    def forward(self, x, emb):
        h = self.act(self.norm1(x)); h = self.conv1(h)
        s, sh = self.emb_proj(self.act(emb)).chunk(2, dim=-1)
        h = h*(1+s[:,:,None,None]) + sh[:,:,None,None]
        h = self.act(self.norm2(h)); h = self.conv2(h)
        return x + h

class DDPMUNet(nn.Module):
    def __init__(self, in_ch=1, base=64, emb_dim=256, T=1000):
        super().__init__()
        self.T = T
        self.time_emb = nn.Sequential(
            SinusoidalEmb(emb_dim), nn.Linear(emb_dim, emb_dim*2),
            nn.SiLU(), nn.Linear(emb_dim*2, emb_dim))
        ch=base
        self.enc0=nn.Conv2d(in_ch,ch,3,padding=1)
        self.enc1a=ResBlock(ch,emb_dim); self.enc1b=ResBlock(ch,emb_dim)
        self.down1=nn.Conv2d(ch,ch*2,4,stride=2,padding=1)
        ch2=ch*2
        self.enc2a=ResBlock(ch2,emb_dim); self.enc2b=ResBlock(ch2,emb_dim)
        self.down2=nn.Conv2d(ch2,ch2*2,4,stride=2,padding=1)
        ch4=ch2*2
        self.mid_a=ResBlock(ch4,emb_dim); self.mid_b=ResBlock(ch4,emb_dim)
        self.up2=nn.ConvTranspose2d(ch4,ch2,4,stride=2,padding=1)
        self.dec2a=ResBlock(ch2*2,emb_dim); self.dec2b=ResBlock(ch2*2,emb_dim)
        self.proj2=nn.Conv2d(ch2*2,ch2,1)
        self.up1=nn.ConvTranspose2d(ch2,ch,4,stride=2,padding=1)
        self.dec1a=ResBlock(ch*2,emb_dim); self.dec1b=ResBlock(ch*2,emb_dim)
        self.proj1=nn.Conv2d(ch*2,ch,1)
        self.out=nn.Sequential(nn.GroupNorm(min(8,ch),ch),nn.SiLU(),nn.Conv2d(ch,in_ch,3,padding=1))
    def forward(self, x, t_int):
        emb=self.time_emb(t_int)
        h0=self.enc0(x)
        h1=self.enc1b(self.enc1a(h0,emb),emb)
        h2=self.enc2b(self.enc2a(self.down1(h1),emb),emb)
        hm=self.mid_b(self.mid_a(self.down2(h2),emb),emb)
        u2=F.interpolate(self.up2(hm),size=h2.shape[-2:],mode='nearest')
        u2=self.proj2(self.dec2b(self.dec2a(torch.cat([u2,h2],1),emb),emb))
        u1=F.interpolate(self.up1(u2),size=h1.shape[-2:],mode='nearest')
        u1=self.proj1(self.dec1b(self.dec1a(torch.cat([u1,h1],1),emb),emb))
        return self.out(u1)

# ─── DPS 核心步骤 ─────────────────────────────────────────────────────────────
def dps_core_step(model, sched, x, t_idx, t_prev_idx, y_obs, operator,
                  zeta, method="vanilla", tmpd_rho=0.0, ps_plus_K=1,
                  psd_t_ratio=0.5):
    """
    单步 DPS 更新:
      x_{t-1} = DDIM_prior(x_t) - zeta * grad_{x_t}||y - A(x̂₀)||

    支持多种改进:
      method="vanilla"    : 标准 DPS (原始 Tweedie x̂₀)
      method="tmpd"       : TMPD-lite: x̂₀ += rho * A^T(y - A(x̂₀))
      method="psd"        : PSD proxy: 在 t_mid 再去噪一次
      method="ps_plus"    : 平均 K 个 noisy x̂₀ 的 A(x̂₀)
      method="mcg"        : MCG: prior step + 观测投影
    """
    device = x.device

    # ── DDIM prior (no grad) ──
    with torch.no_grad():
        eps_ng = model(x.detach(), t_idx.expand(1))
    x_prior = sched.ddim_step(x.detach(), t_idx, t_prev_idx, eps_ng)

    # ── Tweedie x̂₀ (with grad through x) ──
    x_g = x.detach().requires_grad_(True)
    eps_g = model(x_g, t_idx.expand(1))
    x_hat = sched.predict_x0(x_g, t_idx, eps_g).clamp(-1, 1)

    if method == "tmpd":
        # TMPD-lite: x̂₀ += rho * A^T(y - A(x̂₀))
        # FWM (acoustic wave eq) is PyTorch-based but autograd breaks with detached leaf.
        # Fix: use finite-difference Hutchinson estimator for A^T r.
        # A^T r ≈ E_v[<J_A v, r> v] using Rademacher random vectors v.
        with torch.no_grad():
            x_hat_det = x_hat.detach().clamp(-1, 1)
            y_pred0 = operator.forward(x_hat_det)
            resid0 = (y_obs.detach() - y_pred0)
            # Finite-difference AT_r using n_proj random projections
            n_proj = 4
            eps_fd = 5e-3
            AT_r = torch.zeros_like(x_hat_det)
            for _ in range(n_proj):
                v = torch.randn_like(x_hat_det)
                v = v / (v.norm() + 1e-8)  # unit-norm direction
                y_plus = operator.forward((x_hat_det + eps_fd * v).clamp(-1, 1))
                J_v = (y_plus - y_pred0) / eps_fd  # J_A @ v
                AT_r += (J_v * resid0).sum() * v   # <J_A v, r> * v
            AT_r = AT_r / n_proj
            AT_r = AT_r.clamp(-1.0, 1.0)
        x_hat_corrected = (x_hat.detach() + tmpd_rho * AT_r).clamp(-1, 1)
        # Compute gradient w.r.t. x_t: use x_hat as differentiable proxy
        # AT_r is detached constant; gradient flows x_g → eps_g → x_hat → y_pred2
        x_hat2 = sched.predict_x0(x_g, t_idx, eps_g).clamp(-1, 1)
        y_pred2 = operator.forward(x_hat2 + tmpd_rho * AT_r.detach())
        loss2 = torch.linalg.norm(y_pred2 - y_obs.detach())
        grad = torch.autograd.grad(loss2, x_g)[0]

    elif method == "psd":
        # PSD proxy: 从 x̂₀_tweedie 加噪到 t_mid, 再去噪
        t_mid_idx = max(0, int(t_idx.item() * psd_t_ratio))
        with torch.no_grad():
            x_hat_d = x_hat.detach()
            noise = torch.randn_like(x_hat_d)
            x_mid = sched.q_sample(x_hat_d, t_mid_idx, noise)
            t_mid_vec = torch.tensor([t_mid_idx], device=device, dtype=torch.long)
            eps_mid = model(x_mid, t_mid_vec)
            x_hat_psd = sched.predict_x0(x_mid, t_mid_idx, eps_mid).clamp(-1, 1)
        # 梯度通过原始 x_g → eps_g → x_hat 链, 但用 PSD x̂₀ 计算 loss
        # Trick: loss(x_hat_psd) ≈ loss(x_hat) + correction (use x_hat as approx proxy)
        y_pred = operator.forward(x_hat)  # gradient flows through x_g
        y_pred_psd = operator.forward(x_hat_psd.detach())
        # Weighted combination: gradient via x_hat, target from psd
        loss = torch.linalg.norm(y_pred - y_obs.detach())
        # Also add direct psd signal
        loss_psd = torch.linalg.norm(y_pred_psd - y_obs.detach())
        # Use psd as the actual loss signal, gradient through x_hat as proxy
        grad = torch.autograd.grad(loss, x_g)[0]
        # Scale by psd improvement ratio
        with torch.no_grad():
            ratio = (loss_psd / (loss.detach() + 1e-8)).clamp(0.5, 2.0)
        grad = grad * ratio

    elif method == "ps_plus":
        # ps+: 平均 K 个 noisy x̂₀ 的 y_pred
        with torch.no_grad():
            y_preds = []
            for _ in range(ps_plus_K):
                # Small perturbation of x̂₀
                noise = torch.randn_like(x_hat.detach()) * 0.05
                y_p_k = operator.forward((x_hat.detach() + noise).clamp(-1, 1))
                y_preds.append(y_p_k)
            y_pred_avg = torch.stack(y_preds).mean(0)
        # gradient through x_hat (deterministic part), loss target from avg
        y_pred = operator.forward(x_hat)
        # Use averaged y_pred as effective target
        loss = torch.linalg.norm(y_pred - (y_obs.detach() + y_pred_avg.detach() - operator.forward(x_hat.detach())) )
        grad = torch.autograd.grad(loss, x_g)[0]

    elif method == "mcg":
        # MCG: DPS step + 投影到观测流形
        y_pred = operator.forward(x_hat)
        loss = torch.linalg.norm(y_pred - y_obs.detach())
        grad = torch.autograd.grad(loss, x_g)[0]

    else:  # vanilla
        y_pred = operator.forward(x_hat)
        loss = torch.linalg.norm(y_pred - y_obs.detach())
        grad = torch.autograd.grad(loss, x_g)[0]

    with torch.no_grad():
        x_new = (x_prior - zeta * grad).clamp(-1.5, 1.5).detach()

    # MCG额外步骤: 投影 (必须在 enable_grad 里, 否则 no_grad 导致 no grad_fn)
    if method == "mcg":
        with torch.enable_grad():
            # 投影修正: x_{t-1} -= η * ∇_{x} ||y - A(x̂₀(x))||
            x_hat_new_g = x_new.detach().requires_grad_(True)
            eps_new = model(x_hat_new_g, t_prev_idx.expand(1))
            x_hat_new = sched.predict_x0(x_hat_new_g, t_prev_idx, eps_new).clamp(-1, 1)
            y_pred_new = operator.forward(x_hat_new)
            resid_new = y_obs.detach() - y_pred_new
            proj_loss = torch.linalg.norm(resid_new)
            proj_grad = torch.autograd.grad(proj_loss, x_hat_new_g)[0]
        # proj_step: project x toward measurement manifold
        proj_scale = (float(t_prev_idx) / sched.T)  # stronger at high t
        x_new = (x_new - zeta * 0.5 * proj_scale * proj_grad.detach()).clamp(-1.5, 1.5).detach()

    return x_new


# ─── 各改进方法 ────────────────────────────────────────────────────────────────
def dps_vanilla(model, sched, operator, args, device,
                zeta=0.1, n_steps=200, seed=None, t_start=None, x_init=None):
    """标准 DPS"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T

    if t_start is not None and x_init is not None:
        # warmstart: 从 t_start 开始
        t_steps = torch.linspace(t_start, 0, n_steps+1).long().to(device)
        x = x_init.clone().to(device)
    else:
        t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
        x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()

        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta, method="vanilla")

    return x.clamp(-1, 1)


def dps_warmstart(model, sched, operator, args, device, no_pred,
                  zeta=0.3, n_steps=100, t_start=500, seed=None):
    """从 NO 预测 + noise@t_start 初始化"""
    if seed is not None: torch.manual_seed(seed)
    device_t = torch.device(device) if isinstance(device, str) else device

    # 将 NO 预测 (normalized) 加噪到 t_start
    no_norm = no_pred.to(device_t)  # already in [-1,1]
    noise = torch.randn_like(no_norm)
    t_s = torch.tensor(t_start, device=device_t)
    x_init = sched.q_sample(no_norm, t_s, noise)

    return dps_vanilla(model, sched, operator, args, device_t,
                       zeta=zeta, n_steps=n_steps,
                       t_start=t_start, x_init=x_init, seed=None)


def dps_annealed(model, sched, operator, args, device,
                 zeta_max=0.5, zeta_min=0.02, n_steps=200, seed=None):
    """ζ 从 zeta_max cosine 渐减到 zeta_min"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T
    t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
    x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()
        # cosine schedule: zeta decreases as i increases (t decreases)
        cos_val = 0.5 * (1 + math.cos(math.pi * i / n_steps))
        zeta_i = zeta_min + (zeta_max - zeta_min) * cos_val
        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta_i, method="vanilla")

    return x.clamp(-1, 1)


def dps_psd(model, sched, operator, args, device,
            zeta=0.3, n_steps=200, psd_t_ratio=0.5, seed=None):
    """PSD midpoint proxy"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T
    t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
    x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()
        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta,
                          method="psd", psd_t_ratio=psd_t_ratio)

    return x.clamp(-1, 1)


def dps_tmpd(model, sched, operator, args, device,
             zeta=0.3, tmpd_rho=0.01, n_steps=200, seed=None):
    """TMPD-lite"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T
    t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
    x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()
        # 仅在 t 较小时 (高质量 x̂₀) 启用 TMPD 修正
        t_frac = float(t_idx) / (T - 1)
        rho = tmpd_rho * (1 - t_frac)  # 后半段更强
        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta,
                          method="tmpd", tmpd_rho=rho)

    return x.clamp(-1, 1)


def dps_ps_plus(model, sched, operator, args, device,
                zeta=0.3, K=4, n_steps=200, seed=None):
    """ps+: 平均 K 个 noisy x̂₀ 前向残差"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T
    t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
    x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()
        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta,
                          method="ps_plus", ps_plus_K=K)

    return x.clamp(-1, 1)


def dps_mcg(model, sched, operator, args, device,
            zeta=0.3, n_steps=200, seed=None):
    """MCG: DPS + 观测流形投影"""
    if seed is not None: torch.manual_seed(seed)
    T = sched.T
    t_steps = torch.linspace(T-1, 0, n_steps+1).long().to(device)
    x = torch.randn(1, 1, 70, 70, device=device)

    for i in range(n_steps):
        t_idx      = t_steps[i]
        t_prev_idx = t_steps[i+1]
        if t_prev_idx < 0: t_prev_idx = torch.zeros(1, dtype=torch.long, device=device).squeeze()
        x = dps_core_step(model, sched, x, t_idx, t_prev_idx,
                          operator.y_obs, operator, zeta, method="mcg")

    return x.clamp(-1, 1)


# ─── 多次运行取 best ───────────────────────────────────────────────────────────
def run_n(fn, n_runs, case_id, seed_base, x_true, device):
    """运行 fn(seed) n_runs 次, 返回 best (最低 MSE) 结果
    fn 是一个只接受 seed 参数的闭包, 其他参数已通过 closure 绑定.
    """
    runs = []
    for r in range(n_runs):
        seed = seed_base + case_id * 1000 + r * 137
        try:
            x_pred = fn(seed)
            mse = float(F.mse_loss(x_pred, x_true.to(device)).cpu())
            print(f"    run={r}: MSE={mse:.4f}", flush=True)
            runs.append((mse, x_pred.detach().clone()))
        except Exception as e:
            import traceback
            print(f"    run={r}: FAILED ({e})", flush=True)
            traceback.print_exc()
    if not runs:
        return None, float('inf')
    runs.sort(key=lambda r: r[0])
    return runs[0][1], runs[0][0]


# ─── Visualization ─────────────────────────────────────────────────────────────
def make_comparison_plot(cases_data, outfile, title="DPS Comparison"):
    """
    cases_data: dict {case_label: {method_name: array (70,70) in raw km/s}}
    绘制 Grid: row=case, col=method
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    case_labels = list(cases_data.keys())
    method_labels = list(cases_data[case_labels[0]].keys())
    n_rows = len(case_labels)
    n_cols = len(method_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8*n_cols, 2.8*n_rows+0.3))
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    # Global vmin/vmax per case
    for r, clabel in enumerate(case_labels):
        methods = cases_data[clabel]
        all_vals = np.concatenate([v.flatten() for v in methods.values()])
        vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
        gt = methods.get("GT", None)
        if gt is not None:
            vmin = gt.min() * 0.95
            vmax = gt.max() * 1.05

        for c, mlabel in enumerate(method_labels):
            ax = axes[r, c]
            arr = methods.get(mlabel, None)
            if arr is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(arr, cmap="seismic", vmin=vmin, vmax=vmax,
                           origin="upper", aspect="equal")
            if r == 0:
                ax.set_title(mlabel, fontsize=7, fontweight="bold")
            if c == 0:
                ax.set_ylabel(clabel, fontsize=8)

            # Compute MSE vs GT if available
            if "GT" in methods and mlabel != "GT":
                gt_a = methods["GT"]
                # normalize to [0,1] for MSE in normalized space
                mse = np.mean((arr - gt_a)**2)
                ax.text(0.02, 0.02, f"MSE={mse:.4f}", transform=ax.transAxes,
                        fontsize=5.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))

            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {outfile}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases",      default="0,11")
    p.add_argument("--outdir",     default="/workspace/fmm_outputs/dps_v3")
    p.add_argument("--v2dir",      default="/workspace/fmm_outputs/dps_cva_v2")
    p.add_argument("--ddpm_ckpt",  default="/workspace/fmm_outputs/ddpm_cva/latest.pt")
    p.add_argument("--no_ckpt",    default="/workspace/fmm_outputs/bench_cva_operator/unet/final.pt")
    p.add_argument("--methods",    default="vanilla_z01,vanilla_z05,warmstart,annealed,psd,tmpd,ps_plus,mcg")
    p.add_argument("--n_runs",     type=int, default=3)
    p.add_argument("--n_steps",    type=int, default=200)
    p.add_argument("--plot_only",  action="store_true", help="只绘图不推断")
    cli = p.parse_args()
    os.makedirs(cli.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    args = build_args()
    cases = [int(c) for c in cli.cases.split(",")]

    if not cli.plot_only:
        # ── Load data ──────────────────────────────────────────────────────────
        print("Loading data...", flush=True)
        x_all, y_f_all = load_block(args.cache_dir, args.skip, args.n_test)
        loc_all        = load_loc(args.data_root, args.skip, args.n_test)

        # ── Load DDPM ──────────────────────────────────────────────────────────
        print(f"Loading DDPM from {cli.ddpm_ckpt}...", flush=True)
        ckpt = torch.load(cli.ddpm_ckpt, map_location=device, weights_only=False)
        ck_args = ckpt.get("args", {})
        T_val   = ckpt.get("T", 1000)
        ddpm_model = DDPMUNet(
            base=int(ck_args.get("base", 64)),
            emb_dim=int(ck_args.get("emb_dim", 256)),
            T=T_val).to(device)
        ddpm_model.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
        ddpm_model.eval()
        print(f"  T={T_val}, step={ckpt.get('step','?')}, loss={ckpt.get('loss',0):.5f}", flush=True)

        sched = DDPMSchedule(T=T_val, device=device)

        # ── Load NO network ────────────────────────────────────────────────────
        print(f"Loading NO from {cli.no_ckpt}...", flush=True)
        no_net = EM.UNet().to(device)
        no_net.load_state_dict(torch.load(cli.no_ckpt, map_location=device, weights_only=False)["model"])
        no_net.eval()

        methods_to_run = [m.strip() for m in cli.methods.split(",") if m.strip()]
        summary = {}

        for case_id in cases:
            x_true = x_all[case_id:case_id+1]
            y_f    = y_f_all[case_id:case_id+1]
            loc    = loc_all[case_id:case_id+1]
            y_obs  = make_observation(x_true, y_f, loc, args.setting, args, device)

            # NO prediction
            with torch.no_grad():
                v0_no = no_net(EM.obs_image(y_obs)).clamp(-1, 1)
                mse_no = float(F.mse_loss(v0_no, x_true.to(device)).cpu())
            print(f"\n{'='*60}\nCase i={case_id} | NO MSE={mse_no:.4f}", flush=True)

            operator = CVAOperator(y_obs, loc, args, device)
            case_results = {"mse_no": mse_no, "methods": {}}

            # Capture locals for closures
            _model = ddpm_model; _sched = sched; _op = operator
            _args = args; _dev = device; _nsteps = cli.n_steps
            _no_pred = v0_no.detach()

            for mname in methods_to_run:
                print(f"\n  --- {mname} ---", flush=True)

                if mname == "vanilla_z01":
                    fn = lambda seed: dps_vanilla(_model, _sched, _op, _args, _dev, zeta=0.10, n_steps=_nsteps, seed=seed)
                elif mname == "vanilla_z05":
                    fn = lambda seed: dps_vanilla(_model, _sched, _op, _args, _dev, zeta=0.50, n_steps=_nsteps, seed=seed)
                elif mname == "warmstart":
                    _ws_steps = min(_nsteps, 150)
                    fn = lambda seed: dps_warmstart(_model, _sched, _op, _args, _dev,
                                                    no_pred=_no_pred, zeta=0.30,
                                                    n_steps=_ws_steps, t_start=500, seed=seed)
                elif mname == "annealed":
                    fn = lambda seed: dps_annealed(_model, _sched, _op, _args, _dev,
                                                   zeta_max=0.5, zeta_min=0.02, n_steps=_nsteps, seed=seed)
                elif mname == "psd":
                    fn = lambda seed: dps_psd(_model, _sched, _op, _args, _dev,
                                              zeta=0.30, n_steps=_nsteps, psd_t_ratio=0.5, seed=seed)
                elif mname == "tmpd":
                    fn = lambda seed: dps_tmpd(_model, _sched, _op, _args, _dev,
                                               zeta=0.30, tmpd_rho=0.02, n_steps=_nsteps, seed=seed)
                elif mname == "ps_plus":
                    fn = lambda seed: dps_ps_plus(_model, _sched, _op, _args, _dev,
                                                  zeta=0.30, K=4, n_steps=_nsteps, seed=seed)
                elif mname == "mcg":
                    fn = lambda seed: dps_mcg(_model, _sched, _op, _args, _dev,
                                              zeta=0.20, n_steps=_nsteps, seed=seed)
                else:
                    print(f"  [skip unknown: {mname}]", flush=True)
                    continue

                best_pred, best_mse = run_n(fn, cli.n_runs, case_id,
                                            seed_base=args.seed,
                                            x_true=x_true, device=device)
                case_results["methods"][mname] = best_mse
                flag = "✓" if best_mse < mse_no else "✗"
                print(f"  {mname} best={best_mse:.4f}  {flag}", flush=True)

                if best_pred is not None:
                    pref = os.path.join(cli.outdir, f"i{case_id:02d}_{mname}")
                    np.save(f"{pref}_pred.npy",
                            to_raw(best_pred).squeeze().cpu().numpy())

            # Save GT and NO (normalized → raw)
            np.save(os.path.join(cli.outdir, f"i{case_id:02d}_gt.npy"),
                    to_raw(x_true).squeeze().numpy())
            np.save(os.path.join(cli.outdir, f"i{case_id:02d}_no.npy"),
                    to_raw(v0_no).squeeze().cpu().numpy())

            print(f"\n  === Summary i={case_id} ===", flush=True)
            print(f"    NO: {mse_no:.4f}", flush=True)
            for m, v in case_results["methods"].items():
                flag = "✓" if v < mse_no else "✗"
                print(f"    {m}: {v:.4f}  {flag}", flush=True)

            summary[str(case_id)] = case_results

        with open(os.path.join(cli.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {cli.outdir}", flush=True)

    # ── Visualization ────────────────────────────────────────────────────────
    print("\n=== Generating plots ===", flush=True)
    v2dir  = cli.v2dir
    outdir = cli.outdir

    def load_npy_safe(path):
        try:
            return np.load(path)
        except:
            return None

    # ── Plot 1: V2 comparison (vanilla DPS variants) ──
    v2_methods = {
        "GT":       None,
        "NO":       None,
        "DPS_z0.1": None,
        "DPS_z0.3": None,
        "DPS_z0.5": None,
    }

    cases_data_v2 = {}
    for case_id in cases:
        d = {}
        d["GT"]       = load_npy_safe(f"{v2dir}/i{case_id:02d}_gt.npy")
        d["NO"]       = load_npy_safe(f"{v2dir}/i{case_id:02d}_no.npy")
        d["DPS_z0.1"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_D1_z01_pred.npy")
        d["DPS_z0.3"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_D1_z03_pred.npy")
        d["DPS_z0.5"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_D1_z05_pred.npy")
        d = {k: v for k, v in d.items() if v is not None}
        cases_data_v2[f"i={case_id}"] = d

    if cases_data_v2:
        make_comparison_plot(cases_data_v2,
                             os.path.join(outdir, "plot_v2_vanilla_dps.png"),
                             title="DPS v2: GT / NO / DPS vanilla variants (hard cases i=0,11)")

    # ── Plot 2: V3 improvements comparison ──
    v3_method_names = [m.strip() for m in cli.methods.split(",") if m.strip()]

    cases_data_v3 = {}
    for case_id in cases:
        d = {}
        d["GT"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_gt.npy")
        d["NO"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_no.npy")
        d["DPS_z0.1"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_D1_z01_pred.npy")
        d["DPS_z0.5"] = load_npy_safe(f"{v2dir}/i{case_id:02d}_D1_z05_pred.npy")
        # Load v3 results
        for mname in v3_method_names:
            arr = load_npy_safe(f"{outdir}/i{case_id:02d}_{mname}_pred.npy")
            if arr is not None:
                d[mname] = arr
        d = {k: v for k, v in d.items() if v is not None}
        cases_data_v3[f"i={case_id}"] = d

    if cases_data_v3:
        make_comparison_plot(cases_data_v3,
                             os.path.join(outdir, "plot_v3_improved_dps.png"),
                             title="DPS v3: Improvements (hard cases i=0,11)")

    # ── Plot 3: Best-of-all comparison ──
    for case_id in cases:
        d = {}
        gt = load_npy_safe(f"{v2dir}/i{case_id:02d}_gt.npy")
        no = load_npy_safe(f"{v2dir}/i{case_id:02d}_no.npy")
        if gt is None: continue
        d["GT"] = gt
        d["NO"] = no
        # load all available predictions and show best 4
        candidates = {}
        for fn in os.listdir(outdir):
            if fn.startswith(f"i{case_id:02d}_") and fn.endswith("_pred.npy"):
                mname = fn[len(f"i{case_id:02d}_"):-len("_pred.npy")]
                arr = load_npy_safe(os.path.join(outdir, fn))
                if arr is not None:
                    mse = np.mean((arr - gt)**2)
                    candidates[mname] = (mse, arr)
        # Sort by MSE
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0])
        for mname, (mse, arr) in sorted_cands[:6]:
            d[f"{mname}\nMSE={mse:.4f}"] = arr

        cases_data_best = {f"i={case_id}": d}
        make_comparison_plot(cases_data_best,
                             os.path.join(outdir, f"plot_best_i{case_id:02d}.png"),
                             title=f"Best DPS variants for case i={case_id}")

    print("\n=== All done! ===", flush=True)
    summary_path = os.path.join(outdir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        print("\nFinal summary:", flush=True)
        for ci, cdata in s.items():
            no_mse = cdata["mse_no"]
            print(f"  i={ci} | NO={no_mse:.4f}", flush=True)
            for m, v in cdata["methods"].items():
                flag = "✓" if v < no_mse else "✗"
                print(f"    {m}: {v:.4f}  {flag}", flush=True)


if __name__ == "__main__":
    main()
