#!/bin/bash
# CPU recordings (no GPU) of the 10 video scenes for two modes, 3 at a time, then software-EGL renders.
cd /home/konstantinsmirnov/robotics/cat-energy-improve
export MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json OMP_NUM_THREADS=4
export CAT_PATH_MAP="/l/users/konstantin.smirnov/cat_root=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab"
P=/tmp/cat-energy-test-env/bin/python; B=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1
V=/tmp/cat-improve/videos; CK=/tmp/cat-energy-current-videos/inputs/COMBO_FIX_315808_u150.pt
mapfile -t SC < <($P -c "import json;[print(v) for v in json.load(open('$V/../video_scenes.json')).values()]")
jobs_=()
for mode in none plan; do
  for s in "${SC[@]}"; do
    mix=none=1; [[ $mode == plan ]] && mix=plan=1
    out=$V/rec_${mode}_$s.npz
    [[ -s $out ]] && continue
    $P -u scripts/record_plan_reflex.py $CK --device cpu --scenes $s --mix $mix --speed 0.15 --steps 6000 --bank $B \
      --goal-seed-cache /tmp/cat-plan-blind/goal-cache --planner-checkpoint /tmp/cat-plan-blind/planner.pt --out $out \
      > $V/rec_${mode}_$s.log 2>&1 &
    while [[ $(jobs -rp | wc -l) -ge 3 ]]; do sleep 5; done
  done
done
wait
grep -h RECORDED $V/rec_*.log | cut -c1-260
