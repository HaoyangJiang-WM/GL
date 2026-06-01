#!/usr/bin/env python3
"""
gen_fwi_obs.py — parallel seismic observation caching for FVB dataset
======================================================================
Each pod runs ONE shard.  Results are saved as:
    /data10/fwi_cache/fvb/vel_{i:04d}.pt    velocity  [500,1,70,70]  float32 norm
    /data10/fwi_cache/fvb/seis_{i:04d}.pt   seismic   [500,5,T,70]  float32 norm

Usage (on pod, background):
    python gen_fwi_obs.py --shard 0  --n_shards 12 \
        --vel_root /workspace/fdo-fwi/data/fvb/velocity \
        --out_dir  /data10/fwi_cache/fvb

Launch all 12 shards from jump host:
    for i in $(seq 0 11); do
      POD=flow${i}-xxx
      kubectl exec $POD -n hjiang16 -- bash -c "nohup python /data10/code/gen_fwi_obs.py \
        --shard $i --n_shards 12 > /tmp/gen_shard${i}.log 2>&1 &"
    done
"""

import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/fdo-fwi/data/fvb")
from data_gen_f import FWM


def vel_norm(x):
    x = torch.from_numpy(x.astype("float32"))
    if x.shape[-2:] != (70, 70):
        x = F.interpolate(x, (70, 70), mode="bilinear", align_corners=False)
    x = x.clamp(1500, 4500)
    return (x - 1500.0) / 3000.0 * 2.0 - 1.0


def to_raw(x_norm):
    return (x_norm.clamp(-1.2, 1.2) + 1.0) * 1500.0 + 1500.0


def pad_vel(v, nbc):
    return F.pad(v, (nbc, nbc, nbc, nbc), mode="replicate")


def seismic_norm(seis):
    def log_t(x):
        return torch.log1p(torch.abs(x)) * torch.sign(x)
    lo = log_t(torch.tensor(-30.0))
    hi = log_t(torch.tensor(60.0))
    return (log_t(seis) - lo) / (hi - lo + 1e-6) * 2.0 - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard",    type=int, default=0)
    p.add_argument("--n_shards", type=int, default=12)
    p.add_argument("--vel_root", default="/workspace/fdo-fwi/data/fvb/velocity")
    p.add_argument("--out_dir",  default="/data10/fwi_cache/fvb")
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--nt",       type=int, default=300)
    p.add_argument("--sr",       type=int, default=2)   # sampling_rate
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FWM geometry (matches run_alsd_fwi.py)
    nbc = 120; dx = 10; dt = 1e-3; freq = 15.0; sz = 10; gz = 10
    grids = 70
    sx = np.linspace(0, grids - 1, 5)    * dx
    gx = np.linspace(0, grids - 1, grids) * dx

    all_files = sorted(glob.glob(os.path.join(args.vel_root, "model*.npy")))
    # assign files to this shard
    my_files = [f for i, f in enumerate(all_files) if i % args.n_shards == args.shard]
    print(f"shard {args.shard}/{args.n_shards}: {len(my_files)} files", flush=True)

    for fpath in my_files:
        idx = int(os.path.splitext(os.path.basename(fpath))[0].replace("model", ""))
        vel_out  = os.path.join(args.out_dir, f"vel_{idx:04d}.pt")
        seis_out = os.path.join(args.out_dir, f"seis_{idx:04d}.pt")
        if os.path.exists(vel_out) and os.path.exists(seis_out):
            print(f"  skip {idx} (already cached)", flush=True)
            continue

        raw = np.load(fpath)                # [500,1,70,70]
        x   = vel_norm(raw).to(device)      # normalised float32 [500,1,70,70]
        N   = x.shape[0]

        seises = []
        t0 = time.time()
        for i in range(0, N, args.batch):
            xb = x[i:i + args.batch]
            v  = pad_vel(to_raw(xb), nbc)
            s  = FWM(v, nbc, dx, args.nt, dt, freq, sx, sz, gx, gz, args.sr)
            seises.append(seismic_norm(s).cpu())

        vel_cpu  = x.cpu()
        seis_cpu = torch.cat(seises, 0)     # [500, 5, T, 70]
        torch.save(vel_cpu,  vel_out)
        torch.save(seis_cpu, seis_out)
        elapsed = time.time() - t0
        print(json.dumps({"file": idx, "vel": list(vel_cpu.shape),
                          "seis": list(seis_cpu.shape), "sec": round(elapsed, 1)}),
              flush=True)

    print(f"shard {args.shard} done", flush=True)


if __name__ == "__main__":
    main()
