# Source inside a job: builds the job's working copy on node-local disk and cds into it.
#   $WORK/{code from ~/Click-and-Traverse-Mjlab, data/ unpacked from the tar, outputs -> /l/users/$USER/outputs}
L=/l/users/$USER
WORK=$TMPDIR/cat_work
mkdir -p "$WORK" "$L/outputs"
rsync -a --exclude .git --exclude outputs --exclude data "$HOME/Click-and-Traverse-Mjlab/" "$WORK/"
mkdir -p "$WORK/data"
tar -xf "$L/data/cat_data_v1.tar" -C "$WORK"
rsync -a "$HOME/Click-and-Traverse-Mjlab/outputs/" "$L/outputs/"     # small inputs (SGS priors, routing, feasibility)
ln -sfn "$L/outputs" "$WORK/outputs"
[[ -e "$L/outputs/saved_checkpoints" ]] || ln -sfn "$L/checkpoints" "$L/outputs/saved_checkpoints"
cd "$WORK"
export REPO=$WORK
echo "staged $(du -sh $WORK/data | cut -f1) data in $WORK on $(hostname)"
