#!/bin/bash
# Software-EGL renders (no GPU) of the CPU recordings into the project's improve-videos folder.
cd /home/konstantinsmirnov/robotics/cat-energy-improve
export MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export CAT_PATH_MAP="/l/users/konstantin.smirnov/cat_root=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab"
P=/tmp/cat-energy-test-env/bin/python; B=/home/konstantinsmirnov/robotics/Click-and-Traverse-Mjlab/data/furniture/tablegap_v1
V=/tmp/cat-improve/videos; OUT=/home/konstantinsmirnov/.local/share/openresearch/files/humanoid-protection/improve-videos
mkdir -p $OUT
for mode in none plan; do
  label="no reference (same network)"; [[ $mode == plan ]] && label="planner-reflex"
  $P - "$V/../video_scenes.json" "$mode" "$label" > $V/map_$mode.json <<'PY'
import json, sys
scenes = json.load(open(sys.argv[1])); mode, label = sys.argv[2], sys.argv[3]
tag = "no-reference" if mode == "none" else "planner-reflex"
print(json.dumps({sid: dict(file=f"{tag}-{i:02d}-" + "-".join(t.replace(",", "").split()),
                            title=f"{t} | {label}") for i, (t, sid) in enumerate(scenes.items())}))
PY
  for f in $V/rec_${mode}_*.npz; do
    $P -u scripts/render_plan_reflex.py --rec $f --out $OUT --bank $B --furnished --scene-map $V/map_$mode.json --max-seconds 40 \
      >> $V/render_$mode.log 2>&1 || echo "render failed $f"
  done
done
for f in $OUT/*.mp4; do echo "$(basename $f) $(ffprobe -v error -show_entries stream=codec_name,width,height,pix_fmt -show_entries format=duration -of compact=p=0:nk=1 $f | tr '\n' ' ')"; done
