#!/usr/bin/env bash
set -euo pipefail

python scripts/run_spapot_loss_search_round.py \
  --preset r14 \
  --round-name round14_classifier_nomass_hybrid \
  --only r14_typeprior_nomass_state7030_spatial020 \
  --device mps \
  --epochs 120 \
  --sample-size 384 \
  --steps-per-interval 8

