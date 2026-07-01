# ICTPolarReal

Official code release for **ICTPolarReal: A Polarized Reflection and Material
Dataset of Real World Objects**.

ICTPolarReal is a CVPR 2026 dataset and benchmark for real-world polarized
reflectance. It contains multi-view cross/parallel polarization captures and
material annotations for material decomposition, relighting, and reconstruction
research.

- Project page: https://jingyangcarl.github.io/ICTPolarReal/
- Paper: https://arxiv.org/abs/2603.24912
- Sample data: https://drive.google.com/drive/u/1/folders/1J2lfWe8rO1ZXpbeVW68u2RSqOocCs-S6

## Quick Start

Clone the repo:

```bash
git clone https://github.com/jingyangcarl/ICTPolarReal.git
cd ICTPolarReal
```

Run the full sample workflow:

```bash
bash run.sh all
```

No manual Python setup is needed for the default path. The training stages use
LoRA to fine-tune the RGB2X `rgb-to-x` and `x-to-rgb` diffusion checkpoints;
a CUDA GPU is strongly recommended.

If `data/sample` is missing or incomplete, the script downloads one complete
camera view with all 346 calibrated cross/parallel OLAT pairs. The default
sample is approximately 400 MB. If Google Drive blocks command-line access,
download the sample in a browser, place it under `data/sample`, and rerun the
same command.

## What `run.sh all` Does

| Step | Action | Result |
| --- | --- | --- |
| 1 | Set up the environment | Creates or reuses the `ictpolarreal` environment and installs the package. |
| 2 | Check Python packages | Verifies imports and reports PyTorch/CUDA availability. |
| 3 | Prepare sample data | Validates `data/sample`; if needed, downloads one complete 346-light camera view. |
| 4 | Decompose polarization data | Fits diffuse normals/albedo and specular BRDF parameters, then writes material PNG maps. |
| 5 | Fine-tune RGB2X | Trains inverse and forward models, periodically comparing pretrained and fine-tuned predictions. |
| 6 | Evaluate predictions | Writes CSV metrics and a JSON summary under `outputs/`. |

## Expected Data Layout

Each object should contain camera folders like this:

```text
data/sample/
  object_name/
    cam00/
      static.exr
      static_cross.exr
      static_parallel.exr
      mask.png
      albedo.exr
      cross/000002.exr ... 000347.exr
      parallel/000002.exr ... 000347.exr
```

The original 350-frame capture layout reserves frames `000000`, `000001`,
`000348`, and `000349` as indicators. They are excluded automatically. The
bundled LSX calibration maps valid frames `000002` through `000347` to light
directions. `run.sh process` also accepts normalized 346-frame sequences.

## Outputs

Default outputs are written to `outputs/`:

- `outputs/material_acquisition/`: decomposed material PNG maps under `<object>/<camera>/brdf/`.
- `outputs/train/inverse/`: prompt-conditioned RGB-to-PBR/polarization LoRA and predictions.
- `outputs/train/forward/gbuffer/`: PBR G-buffer-to-RGB LoRA and relighting predictions.
- `outputs/train/forward/polarization/`: cross/parallel-to-RGB LoRA and relighting predictions.
- `outputs/train/*/eval/`: per-step comparison images, CSV metrics, JSON summaries, and metric history.
- `outputs/eval_ictpolarreal_decomposition/`: CSV metrics and JSON summary.

## Flexible Usage

The default command is enough for the sample release. Use options only when
running on a different machine or dataset:

```bash
bash run.sh check-data
bash run.sh process --data-root /path/to/data --output-root /path/to/out
bash run.sh train --data-root /path/to/data --train-stage inverse
bash run.sh train --data-root /path/to/data --train-stage forward --forward-mode gbuffer
bash run.sh evaluate --data-root /path/to/data --pred-root /path/to/predictions
```

Useful options:

- `--data-root PATH`: dataset location. Default: `data/sample`.
- `--output-root PATH`: output location. Default: `outputs`.
- `--torch-variant cpu --device cpu`: use CPU for diagnostics; diffusion training is slow without CUDA.
- `--max-lights N`: use a sphere-wide subset for a quick diagnostic; the default
  346-light fit is recommended for material quality.
- `--backend torch --device cuda`: explicitly select the PyTorch optimizer.
- `--train-stage inverse|forward|both`: choose the training stage.
- `--inverse-workflow pbr|polarization|both`: choose inverse supervision targets.
- `--forward-mode gbuffer|polarization|both`: choose the forward conditioning representation.
- `--train-steps N`: set optimizer steps for each selected model; the default 20-step run is a pipeline check.
- `--train-dry-run`: validate all tensors without loading diffusion checkpoints.
- `--resume latest`: continue from the newest checkpoint in each selected stage.
- `--train-eval-steps N`: periodically compare frozen pretrained and current fine-tuned weights.
- `--train-eval-samples N`: set the fixed evaluation subset size; `0` disables in-training evaluation.
- `--material-root PATH`: use precomputed material maps from another run.
- `--skip-setup`: reuse the current environment.

Objaverse-style evaluation uses `configs/eval_objaverse_samples.json`; see
`samples/objaverse/README.md` for the expected sample layout.

## Manual Install

Use this only if you do not want the shell script to manage the environment:

```bash
conda create -n ictpolarreal python=3.10 -y
conda activate ictpolarreal
pip install -e ".[dev,train]"
```

## Citation

```bibtex
@inproceedings{yang2026ictpolarreal,
  title     = {A Polarized Reflection and Material Dataset of Real World Objects},
  author    = {Yang, Jing and Dharanikota, Krithika and Jia, Emily and Chen, Haiwei and Zhao, Yajie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
