#!/usr/bin/env python3
"""
Generate normalized FWI observations for multiple OpenFWI-style datasets.

Each pod runs one shard. Output format matches the existing FVB cache:
    vel_{k:04d}.pt   [N,1,70,70]
    seis_{k:04d}.pt  [N,5,T,70]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def vel_norm(arr):
    x = torch.from_numpy(arr.astype("float32"))
    if x.ndim == 3:
        x = x[:, None]
    if x.shape[-2:] != (70, 70):
        x = F.interpolate(x, (70, 70), mode="bilinear", align_corners=False)
    x = x.clamp(1500, 4500)
    return (x - 1500.0) / 3000.0 * 2.0 - 1.0


def log_t(x):
    return torch.log1p(torch.abs(x)) * torch.sign(x)


def seismic_norm(seis):
    lo = log_t(torch.tensor(-30.0, device=seis.device, dtype=seis.dtype))
    hi = log_t(torch.tensor(60.0, device=seis.device, dtype=seis.dtype))
    return (log_t(seis) - lo) / (hi - lo + 1e-6) * 2.0 - 1.0


def to_raw(x_norm):
    return (x_norm.clamp(-1.2, 1.2) + 1.0) * 1500.0 + 1500.0


def pad_vel(v, nbc):
    return F.pad(v, (nbc, nbc, nbc, nbc), mode="replicate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fvb", "cva", "cfa", "sta"], required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=12)
    p.add_argument("--data_root", default="/workspace/fdo-fwi/data")
    p.add_argument("--out_root", default="/data10/fwi_cache")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--nt", type=int, default=300)
    p.add_argument("--sr", type=int, default=2)
    p.add_argument("--max_files", type=int, default=0)
    args = p.parse_args()

    sys.path.insert(0, os.path.join(args.data_root, args.dataset))
    from data_gen_f import FWM

    vel_root = os.path.join(args.data_root, args.dataset, "velocity")
    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(vel_root, "*.npy")))
    if args.max_files > 0:
        files = files[: args.max_files]
    my_files = [(i, f) for i, f in enumerate(files) if i % args.n_shards == args.shard]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nbc = 120
    dx = 10
    dt = 1e-3
    freq = 15.0
    sz = 10
    gz = 10
    grids = 70
    sx = np.linspace(0, grids - 1, 5) * dx
    gx = np.linspace(0, grids - 1, grids) * dx

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "shard": args.shard,
                "n_shards": args.n_shards,
                "total_files": len(files),
                "my_files": len(my_files),
                "device": str(device),
            }
        ),
        flush=True,
    )

    for global_idx, path in my_files:
        vel_out = os.path.join(out_dir, f"vel_{global_idx:04d}.pt")
        seis_out = os.path.join(out_dir, f"seis_{global_idx:04d}.pt")
        if os.path.exists(vel_out) and os.path.exists(seis_out):
            print(json.dumps({"skip": global_idx, "file": os.path.basename(path)}), flush=True)
            continue

        raw = np.load(path)
        x = vel_norm(raw).to(device)
        seises = []
        t0 = time.time()
        for i in range(0, x.shape[0], args.batch):
            xb = x[i : i + args.batch]
            s = FWM(
                pad_vel(to_raw(xb), nbc),
                nbc,
                dx,
                args.nt,
                dt,
                freq,
                sx,
                sz,
                gx,
                gz,
                args.sr,
            )
            seises.append(seismic_norm(s).cpu())

        torch.save(x.cpu(), vel_out)
        seis = torch.cat(seises, 0)
        torch.save(seis, seis_out)
        print(
            json.dumps(
                {
                    "idx": global_idx,
                    "file": os.path.basename(path),
                    "vel": list(x.shape),
                    "seis": list(seis.shape),
                    "sec": round(time.time() - t0, 1),
                }
            ),
            flush=True,
        )

    print(json.dumps({"dataset": args.dataset, "shard": args.shard, "done": True}), flush=True)


if __name__ == "__main__":
    main()
