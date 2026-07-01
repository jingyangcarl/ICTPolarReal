# Training

The release includes compact baselines for validating the data path and
experiment protocol. Training has two stages:

- inverse: polarization observations -> material targets.
- forward: material g-buffers -> relit/static image targets.

The dataset loader supports `polarization`, `gbuffer`, and plain `image` modes.
Run preprocessing first so forward training can read `outputs/material_acquisition`.

Inverse decomposition example:

```bash
bash scripts/ictpolarreal.sh train \
  --data-root /path/to/data \
  --train-stage inverse \
  --input-mode polarization \
  --target albedo \
  --train-steps 1000
```

Forward relighting example:

```bash
bash scripts/ictpolarreal.sh train \
  --data-root /path/to/data \
  --train-stage forward \
  --forward-input-mode gbuffer \
  --forward-target static \
  --train-steps 1000
```

For CVPR reproduction runs, record the data split, camera set, target, checkpoint
path, GPU count, command, and output directory.
