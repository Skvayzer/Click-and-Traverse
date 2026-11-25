#!/bin/bash
cd /home/konstantinsmirnov/robotics/cat-energy-improve
export MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CAT_PATH_MAP="/l/users/konstantin.smirnov/cat_root=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab"
P=/tmp/cat-energy-test-env/bin/python; B=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1
V=/tmp/cat-improve/videos; OUT=/home/konstantinsmirnov/.local/share/openresearch/files/humanoid-protection/improve-videos
$P - "$V/../video_scenes.json" > $V/map_arb.json <<'PY'
import json, sys
scenes = json.load(open(sys.argv[1]))
print(json.dumps({sid: dict(file=f"arbitration-{i:02d}-" + "-".join(t.replace(",", "").split()),
                            title=f"{t} | value-guided plan + arbitration") for i, (t, sid) in enumerate(scenes.items())}))
PY
for f in $V/rec_arb_*.npz; do
  sid=$(basename $f .npz); sid=${sid#rec_arb_}
  name=$($P -c "import json;print(json.load(open('$V/map_arb.json'))['$sid']['file'])")
  [[ -s $OUT/$name.mp4 ]] && continue
  mkdir -p $V/tmp_$name
  ( $P -u scripts/render_plan_reflex.py --rec $f --out $V/tmp_$name --bank $B --furnished --scene-map $V/map_arb.json --max-seconds 40 \
      > $V/tmp_$name.log 2>&1 && mv $V/tmp_$name/$name.mp4 $V/tmp_$name/$name.jpg $OUT/ ) &
  while [[ $(jobs -rp | wc -l) -ge 4 ]]; do sleep 5; done
done
wait
