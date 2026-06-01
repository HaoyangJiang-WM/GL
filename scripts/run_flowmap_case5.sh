#!/usr/bin/env bash
set -euo pipefail

# Reproduction launcher for the raw-noise FlowMap inverse setting used in
# the case-5 study. Paths may need adjustment for a different cluster layout.

python src/flowmap_inverse_case5.py \
  --setting loc \
  --flow_ckpt /workspace/fmm_outputs/otfm_prior_cva30k_greedy_sig00/final.pt \
  --no_ckpt /workspace/fmm_outputs/bench_cva_operator/unet/final.pt \
  --cache_dir /data10/fwi_cache/cva \
  --data_root /workspace/fdo-fwi/data/cva \
  --n_test 32 \
  --seed 42 \
  --only_auto \
  --bg_mode zero \
  --no_seeded_frac 0.0 \
  --misfit_mode focused \
  --direct_mute_frac 0.25 \
  --late_weight 2.0 \
  --select best \
  --ode_steps 5 \
  --method proposal_ms_lgfmi_grad \
  --ensemble 2048 \
  --proposal_ms_scales 16,8,4,2,1 \
  --proposal_ms_topk_start 512 \
  --proposal_ms_union_k 512 \
  --proposal_ms_union_views 16,8,4,2 \
  --lgfmi_k 32 \
  --lgfmi_t_mid 0.1 \
  --lgfmi_scale 4 \
  --lgfmi_gate \
  --lgfmi_grad_steps 8 \
  --lgfmi_grad_lr 0.03 \
  --lgfmi_grad_trust 0.05 \
  --lgfmi_grad_bg 0.0 \
  --lgfmi_grad_sgd \
  --select_score_scales 16,8,4,1
