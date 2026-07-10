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

## Evaluation During Training

Every 100 steps by default, the trainer evaluates a fixed subset with explicit
method identities:

| Method | Supported tasks | Execution |
| --- | --- | --- |
| `rgb2x` | All selected inverse/forward tasks | Frozen base checkpoint in the active process. |
| `rgb2x_ictpolarreal` | All selected inverse/forward tasks | Current LoRA checkpoint in the active process. |
| `diffusion_renderer` | Albedo, normal, specular, G-buffer forward | Cached predictions from the separate Cosmos runner. |
| `lotus` | Normal | Cached predictions. |
| `dsine` | Normal | Cached predictions. |

Register cached results with `--eval-baseline METHOD=PATH`. Predictions may be
stored as `<root>/<object>/<camera>/<light>/<task>.png` or, for camera-level
inverse results, `<root>/<object>/<camera>/<task>.png`. `basecolor.png` and
`base_color.png` are accepted for albedo; `forward_rgb.png`, `pred.png`, and
`result.png` are accepted for forward RGB.

```bash
bash run.sh train --train-stage inverse \
  --eval-baseline diffusion_renderer=/path/to/diffusion_renderer \
  --eval-baseline lotus=/path/to/lotus \
  --eval-baseline dsine=/path/to/dsine
```

Missing roots, files, and unsupported method/task pairs are recorded as
`skipped` in `summary.json`. A final comparison runs when training ends. Each
`step-NNNNNN` folder contains metrics, normalized predictions, targets, and
labeled panels under `comparisons/`; `eval/history.jsonl` tracks all runs.

## Selective Runs

Run only inverse decomposition:

```bash
bash run.sh train --train-stage inverse
```

Run one forward representation:

```bash
bash run.sh train --train-stage forward --forward-mode gbuffer
```

The launcher runs 1,000 steps per selected model, prints loss every 10 steps,
and saves every 250 steps. It also writes every update to
`training_history.csv`. For full experiments, set `--train-steps`,
`--batch-size`, `--grad-accum-steps`, and `--checkpointing-steps`; the YAML
files under `configs/` record the 300,000-step defaults. Use `--train-dry-run`
to inspect the dataset contract without downloading model weights. Resume with
`--resume latest`; each checkpoint contains the LoRA adapter, optimizer state,
and global step.

For reproduction runs, record the data split, selected lights, model revision,
LoRA rank, GPU count, command, checkpoint, and output directory.
