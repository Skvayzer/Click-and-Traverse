# Comparison recipes

Pass a JSON file with `train_furniture.py --env-config-json FILE`, or with
`evaluate_furniture.py --env-config FILE`. Training requires an explicit action
dimension flag when it differs from the default 29. All variants retain the
same physical obstacle/robot collision checks and strict success definition.

| Recipe | Training action flag | Difference from full P |
|---|---|---|
| `B1_legs.json` | `--action-dofs 12` | Retrained legs; added actor probe features, hand shaping, prediction and margins disabled |
| `B2_whole_body.json` | `--action-dofs 29` | Whole-body actions; added actor probe features, hand shaping, prediction and margins disabled |
| `B3_hand_protection.json` | `--action-dofs 29` | Hand probes and approach-sensitive shaping; prediction and uncertainty margins disabled |
| `P_oracle.json` | `--action-dofs 29` | Full proposal with simulator geometry |
| `P_corrupted.json` | `--action-dofs 29` | Full proposal with delayed, noisy and partially unknown map packets |
| `A23_no_wrists.json` | `--action-dofs 23` | Waist and arms controlled; six wrist joints held at nominal |
| `A_prediction{0,1}_margin{0,1}.json` | `--action-dofs 29` | Four combinations on the same corrupted-perception settings |
| `H_{nominal,raised,tucked,contextual}.json` | `--action-dofs 29` | Upper actions overridden by a fixed/contextual posture rule |

For **B0**, evaluate the released native checkpoint with `B1_legs.json` and
`--controller B0`; do not train it. The evaluator maps the named environment
observations to the released 162-input policy. The 12/23/29-action variants all
use the common comparison environment; B0 does not reproduce upstream scores
because physical collision geometry, fields and success rules have changed.

For heuristic comparisons with the same original leg controller, pass that
released native checkpoint and one `H_*.json` file. Twelve native actions are
mapped into the 29-action environment, then the environment supplies the
specified upper posture. These presets are static-geometry checks, not validated
dynamic walking controllers.

Privileged critic inputs retain simulator hand/arm information across trained
variants. `probe_features_enabled=false` ablates the added **actor** inputs;
it does not change the critic contract or remove physical hand collision boxes.
Record the complete config and checkpoint lineage when comparing variants.
