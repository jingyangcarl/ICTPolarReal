# Training

ICTPolarReal fine-tunes RGB2X with LoRA, following the inverse and forward
rendering experiments in the paper. Run material acquisition first, or run the
complete workflow:

```bash
bash run.sh all
```

## Model Stages

| Stage | Base checkpoint | Condition | Prediction | Output |
| --- | --- | --- | --- | --- |
| Inverse | `zheng95z/rgb-to-x` | Ordinary RGB | Albedo, camera-space normal, specular, cross, or parallel image selected by text prompt | `outputs/train/inverse` |
| Forward G-buffer | `zheng95z/x-to-rgb` | Albedo, normal, specular, and irradiance | RGB under the sampled light | `outputs/train/forward/gbuffer` |
| Forward polarization | `zheng95z/x-to-rgb` | Canonical cross/parallel images and irradiance | RGB under the sampled light | `outputs/train/forward/polarization` |

The loader creates one all-white sample and one sample per calibrated OLAT pair.
All stages use v-prediction in RGB2X latent space. The inverse model uses target
prompts; both forward models use an empty prompt.

## Selective Runs

Run only inverse decomposition:

```bash
bash run.sh train --train-stage inverse
```

Run one forward representation:

```bash
bash run.sh train --train-stage forward --forward-mode gbuffer
```

For full experiments, set `--train-steps`, `--batch-size`,
`--grad-accum-steps`, and `--checkpointing-steps`. The YAML files under
`configs/` record the paper-scale defaults. Use `--train-dry-run` to inspect the
dataset contract without downloading model weights. Resume an interrupted run
with `--resume latest`; each checkpoint contains the LoRA adapter, optimizer
state, and global step.

For reproduction runs, record the data split, selected lights, model revision,
LoRA rank, GPU count, command, checkpoint, and output directory.
