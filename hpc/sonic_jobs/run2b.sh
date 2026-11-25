#!/bin/bash
# Run 2b on dep-1: warm start from u600, tablegap_v1 (no beam bank), SGS + side-gap curriculum + adaptive soft contacts.
echo "lock acquired $(date -Is)"
SRC="--checkpoint-native /tmp/cat-sonic/latent/inputs/latent_u600.pt --fresh-optimizer --init-action-std 0.3" \
  /tmp/cat-sonic/latent/launch_src_gpu.sh /home/konstantinsmirnov/robotics/cat-energy-local-runs/sonic_latent_run2b_sgs 1400 4096 \
  --unroll-length 16 --batch-size 1024 --num-minibatches 4 --snapshot-interval-updates 100 \
  --sonic-keep-sgs --sonic-soft-until-cat-clean 0.1 > /tmp/cat-sonic/latent/run2b.log 2>&1
echo "exit $? $(date -Is)"
