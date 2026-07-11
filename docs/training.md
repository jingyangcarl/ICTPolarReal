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

A fresh run evaluates step 0, then repeats every 100 steps by default on the
same fixed subset with explicit method identities:

| Method | Supported tasks | Execution |
| --- | --- | --- |
| `rgb2x` (`RGB2X`) | All selected inverse/forward tasks | Frozen base checkpoint in the active process. |
| `ours` (`Ours`) | All selected inverse/forward tasks | Current LoRA checkpoint in the active process. |
| `diffusion_renderer` | Inverse albedo, normal, and specular | Cosmos 7B inverse renderer, precomputed once. |
| `lotus` | Inverse normal | Lotus normal model, precomputed once. |
| `dsine` | Inverse normal | DSINE normal model, precomputed once. |

Before inverse training, `run.sh` executes missing external methods on the same
fixed sample subset. Their model processes exit before RGB2X training starts,
so the GPU memory and dependency environments remain isolated. Results are
cached as `outputs/baselines/<method>/<object>/<camera>/static/<task>.png` and
reused at every evaluation step.

```bash
bash run.sh baselines
bash run.sh train --train-stage inverse
```

Lotus and DSINE use the compatible `lotus` Python environment. Diffusion
Renderer uses its official `cosmos-predict1` environment and 7B inverse
checkpoint. The launcher finds standard Micromamba environments and source
checkouts under `external/` or the parent directory. Use the corresponding
`--*-python` and `--*-repo` options for other layouts. Manual prediction roots
remain supported with repeatable `--eval-baseline METHOD=PATH` options.

A final comparison runs when training ends. Each `step-NNNNNN` folder contains
metrics, normalized predictions, targets, and labeled panels under
`comparisons/`; `eval/history.jsonl` tracks all runs. Unsupported task/method
pairs are explicitly recorded as `skipped` in `summary.json`.

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
