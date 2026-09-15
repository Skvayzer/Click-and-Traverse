# Draft request — not sent

**Title:** Exact paper reproduction: 355-scene assets/manifests and matching training configuration

Hello CAT authors,

We are trying to reproduce the paper's training collection and baseline setup exactly before evaluating our whole-body hand-protection extension.

Section III-B describes 139 cropped 3D-FRONT scenes and 216 procedural scenes. We checked the public repository, Hugging Face dataset history, and the archives linked through Google Drive and Tsinghua Cloud. The available assets and the released generalist's 37-scene configuration do not identify the complete paper collection. We also read the clarification in issue #11 that the 3D-FRONT specialists are not released and that procedural specialists can cover multiple scenes.

Could you provide, or point us to:

1. The exact 139 accepted 3D-FRONT crops: assets or scene/room IDs, object versions, crop transforms, and the accepted split after filtering.
2. The exact 216 procedural scenes: assets or full parameter/seed manifest, generator revision, field settings, and checksums.
3. The paper-version training code/configuration: body measurement locations, observation/action definitions, reward implementation, start/goal sampling, curriculum, PPO/DAgger settings, and specialist-to-scene mapping.
4. The matching teacher/final-generalist checkpoint provenance and evaluation scene manifests.

The available checkpoint configuration names `D8G2L3O2S13`, but we could not find its scene assets in the published downloads. Could those also be provided?

If some artifacts cannot be released, could you identify which exact-reproduction components remain unavailable? We want to label any reconstruction accurately rather than claim it is the original collection.

Thank you.
