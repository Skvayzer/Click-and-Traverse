#!/bin/bash
# Deploy feature/cos-obstacles HEAD to the cluster as ~/cat_worktrees/cos_<sha> (venv linked from an existing clone).
set -euo pipefail
cd ~/robotics/cat-energy-cos
SHA=$(git rev-parse --short=7 HEAD)
B=/tmp/cat-fullpath/cos-$SHA.bundle
BR=$(git rev-parse --abbrev-ref HEAD); git bundle create $B $BR >/dev/null
ssh -S /tmp/cat-fullpath/mbz-ctl -o BatchMode=yes mbz-hpc "export PATH=/usr/local/bin:/usr/bin:/bin; set -euo pipefail; A=\$HOME/cat_worktrees/energy_overnight_1bc21af; D=\$HOME/cat_worktrees/cos_$SHA; P=\$HOME/cat_worktrees/cos-$SHA.bundle; if [ -e \$D ]; then echo \$D; exit 0; fi; cat >| \$P; git clone -q --shared --no-checkout \$A \$D; git -C \$D fetch -q \$P refs/heads/$BR; git -C \$D checkout -q --detach FETCH_HEAD; ln -s \$A/.venv-mjlab \$D/.venv-mjlab; mkdir -p /l/users/\$USER/cos/logs; echo \$D" < $B
