# Training

The first release includes compact baselines for validating the data path and
experiment protocol. Heavier diffusion models from the research workspace should
be ported only after their configs are made path-independent.

Inverse decomposition example:

```bash
bash scripts/ictpolarreal.sh train \
  --data-root /path/to/data \
  --input static \
  --target albedo \
  --train-steps 1000
```

Forward relighting example:

```bash
python -m ictpolarreal.train.forward \
  --data-root /path/to/data \
  --out-dir outputs/forward_static \
  --input albedo \
  --target static
```

For CVPR reproduction runs, record the data split, camera set, target, checkpoint
path, GPU count, command, and output directory.
