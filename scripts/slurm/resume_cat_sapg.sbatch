#!/usr/bin/env bash
#SBATCH --job-name=cat-sapg-resume
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this recipe with sbatch; do not run training over SSH}"
: "${CAT_RUN_DIR:?Export the existing run directory to resume}"
: "${CAT_SOURCE_ROOT:?Export its pinned detached source checkout}"
: "${CAT_SOURCE_COMMIT:?Export the full reviewed source commit hash}"
if [[ ! "$CAT_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo 'CAT_SOURCE_COMMIT must be an exact 40-character lowercase commit hash.' >&2
    exit 2
fi

CAT_REPO_ROOT="${CAT_REPO_ROOT:-/data1/users/konstantin.smirnov/Click-and-Traverse-WholeBody}"
CAT_REPO_ROOT="$(cd "$CAT_REPO_ROOT" && pwd -P)"
cat_source="$(cd "$CAT_SOURCE_ROOT" && pwd -P)"
cat_run="$(cd "$CAT_RUN_DIR" && pwd -P)"
cat_python="$CAT_REPO_ROOT/.venv/bin/python"
test -x "$cat_python"
for cat_file in run.json launch.json status.json wandb.json resume.msgpack; do
    if [[ ! -f "$cat_run/$cat_file" || -L "$cat_run/$cat_file" ]]; then
        echo "Missing or symlinked resume metadata: $cat_run/$cat_file" >&2
        exit 2
    fi
done
if [[ -e "$cat_run/STOP" || -L "$cat_run/STOP" ]]; then
    echo 'STOP exists; resume will not remove a user stop request.' >&2
    exit 2
fi
if [[ "$(git -C "$cat_source" rev-parse HEAD)" != "$CAT_SOURCE_COMMIT" ]]; then
    echo 'Pinned source checkout does not match CAT_SOURCE_COMMIT.' >&2
    exit 2
fi
if git -C "$cat_source" symbolic-ref -q HEAD >/dev/null; then
    echo 'Resume source must be a detached checkout.' >&2
    exit 2
fi

cat_request_stop() {
    # Request completion of the current update and its ordinary full snapshot.
    # No signals are forwarded during model/runtime initialization.
    touch "$cat_run/STOP"
    printf 'Cooperative stop requested for %s\n' "$cat_run"
}
trap cat_request_stop USR1 TERM INT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$cat_source"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${CAT_XLA_MEM_FRACTION:-0.90}"
# CUDA_VISIBLE_DEVICES belongs to Slurm; never override it here.

cd "$cat_source"
printf 'Pinned source: %s\nSource checkout: %s\nResuming: %s\n' \
    "$CAT_SOURCE_COMMIT" "$cat_source" "$cat_run"
"$cat_python" - "$cat_run" "$cat_source" "$CAT_SOURCE_COMMIT" <<'PY' &
import json
import os
from pathlib import Path
import sys


def resume_arguments(launch, directory):
    """Recover all launch choices from the run, without fresh-run defaults."""
    specification = launch["specification"]
    config = specification["config"]
    policy, fine, sapg = config["policy_config"], config["fine_tuning"], config["sapg"]
    if config.get("algorithm") != "sapg" or fine["mode"] != "cat_train_only":
        raise ValueError("This recipe resumes SAPG training-only runs")
    arguments = ["run", "--resume", "--run-dir", str(directory),
                 "--algorithm", config["algorithm"],
                 "--finetuning", fine["mode"], "--profile", fine["profile"],
                 "--num-envs", str(policy["num_envs"]),
                 "--batch-size", str(policy["batch_size"]), "--seed", str(policy["seed"]),
                 "--sapg-num-policies", str(sapg["num_policies"]),
                 "--sapg-embedding-dim", str(sapg["embedding_dim"]),
                 "--bank-manifest", specification["bank_manifest"]]
    for key, flag in (("body_collision_bank", "--body-collision-bank"),
                      ("body_collision_resets", "--body-collision-resets")):
        if specification.get(key):
            arguments += [flag, specification[key]]
    for key in ("mode", "project", "entity"):
        arguments += ["--wandb-" + key, launch["wandb"][key]]
    return arguments


def main():
    directory, source = (Path(value).resolve() for value in sys.argv[1:3])
    expected_commit = sys.argv[3]
    read = lambda name: json.loads((directory / name).read_text())
    launch, run, identity = read("launch.json"), read("run.json"), read("wandb.json")
    stop = directory / "STOP"
    if stop.exists() or stop.is_symlink():
        raise ValueError("STOP exists; refusing to ignore a stop request")
    if not identity.get("initialized") or identity.get("mode") != "online":
        raise ValueError("An established online W&B identity is required")
    for key in ("mode", "project", "entity"):
        if identity[key] != launch["wandb"][key]:
            raise ValueError("Stored W&B identity differs from the original launch")
    sys.path.insert(0, str(source))
    import train_cat_wholebody as launcher
    if Path(launcher.__file__).resolve() != source / "train_cat_wholebody.py":
        raise ValueError("Imported launcher is outside the pinned source checkout")
    code = launcher.code_identity()
    if code["git_commit"] != expected_commit or code != run["code"]:
        raise ValueError("Source commit/content differs from recorded run identity; migrate explicitly first")
    arguments = resume_arguments(launch, directory)
    if launcher.plan(launcher.parser().parse_args(arguments)) != launch["specification"]:
        raise ValueError("Reconstructed resume arguments change the original launch specification")
    print(json.dumps({"resuming_wandb_id": identity["id"], "source": code,
                      "arguments": arguments}), flush=True)
    # The normal launcher validates and restores the complete learner/runtime
    # contract, locks the existing run, and resumes this same W&B identity.
    os.execv(sys.executable, [sys.executable, str(source / "train_cat_wholebody.py"), *arguments])


if __name__ == "__main__":
    main()
PY
cat_learner_pid=$!

# A trapped signal interrupts Bash wait; retain the learner until its normal
# cooperative stop has saved the latest full runtime snapshot.
while true; do
    if wait "$cat_learner_pid"; then
        cat_exit=0
        break
    else
        cat_exit=$?
    fi
    if ! kill -0 "$cat_learner_pid" 2>/dev/null; then
        break
    fi
done
exit "$cat_exit"
