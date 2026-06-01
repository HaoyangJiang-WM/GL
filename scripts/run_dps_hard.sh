#!/usr/bin/env bash
set -euo pipefail

# Diagnostic DPS/DDPM baseline on hard CVA cases.
# This was much slower than the FlowMap proposal-bank method and should be
# treated as a partial diagnostic unless all cases finish.

python /workspace/dps_v3.py \
  --cases 5,7,9 \
  --outdir /workspace/fmm_outputs/dps_hard \
  --methods vanilla_z01,vanilla_z05,warmstart,annealed,psd,tmpd,ps_plus,mcg \
  --n_runs 8 \
  --n_steps 1000
