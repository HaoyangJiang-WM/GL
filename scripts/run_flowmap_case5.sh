#!/usr/bin/env bash
set -euo pipefail

# Reproduction launcher for the case-5 C2F-SVGD-FM setting.
# Paths may need adjustment for a different cluster layout.

python src/c2f_svgd_fm_case5.py \
  --setting loc \
  --flow_ckpt /workspace/fmm_outputs/otfm_prior_cva30k_greedy_sig00/final.pt \
  --no_ckpt /workspace/fmm_outputs/bench_cva_operator/unet/final.pt \
  --cache_dir /data10/fwi_cache/cva \
  --data_root /workspace/fdo-fwi/data/cva \
  --skip 25005 \
  --n_test 1 \
  --only_auto \
  --bg_mode zero \
  --no_seeded_frac 0.0 \
  --method proposal_ms_svgd \
  --select best \
  --robust_select \
  --ensemble 512 \
  --select_score_scales 16,8,4,2,1 \
  --misfit_mode focused \
  --direct_mute_frac 0.30 \
  --late_weight 4.0 \
  --ode_steps 5 \
  --proposal_ms_scales 24,16,12,8,4,2,1 \
  --proposal_ms_topk_start 256 \
  --proposal_ms_union_k 256 \
  --proposal_ms_union_views 24,16,12,8,4,2,1 \
  --proposal_topk 8 \
  --psd_score rank \
  --psd_mix 0.35 \
  --psd_bands 0.0-0.15,0.15-0.35,0.35-1.0 \
  --psd_weights 1.0,0.7,0.25 \
  --svgd_k 16 \
  --svgd_optim adam \
  --svgd_times 0.85,0.70,0.55,0.40,0.25,0.12 \
  --svgd_scales 8,6,4,3,2,1 \
  --svgd_steps 6,10,5,4,4,4 \
  --svgd_lr 0.040,0.036,0.022,0.018,0.012,0.007 \
  --svgd_prior 0.05,0.08,0.15,0.22,0.38,0.58 \
  --svgd_trust 0.045,0.040,0.026,0.020,0.015,0.010 \
  --svgd_gate \
  --svgd_gate_scales 16,8,4,2,1 \
  --svgd_feature 12 \
  --svgd_bw 1.2 \
  --svgd_repulse 0.4 \
  --seed 202671279 \
  --out results/case5_c2f_svgd_fm_run.json \
  --save_pred_dir outputs/case5_c2f_svgd_fm
