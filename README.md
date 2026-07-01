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

No manual Python setup is needed for the default path. If `data/sample` is
missing or incomplete, the script downloads one complete camera view, including
all 346 calibrated cross/parallel OLAT pairs, from the sample Google Drive folder.
The default sample download is approximately 400 MB. If Google Drive blocks
command-line download, open
the sample link in a browser and place the folder under `data/sample`, then
rerun the same command.

## What `run.sh all` Does

| Step | Action | Result |
| --- | --- | --- |
| 1 | Set up the environment | Creates or reuses the `ictpolarreal` environment and installs the package. |
| 2 | Check Python packages | Verifies imports and reports PyTorch/CUDA availability. |
| 3 | Prepare sample data | Validates `data/sample`; if needed, downloads one complete 346-light camera view. |
| 4 | Decompose polarization data | Fits diffuse normals/albedo and specular BRDF parameters, then writes material PNG maps. |
| 5 | Run training smoke jobs | Runs inverse polarization-to-material training and forward g-buffer-to-image training. |
| 6 | Evaluate predictions | Writes CSV metrics and a JSON summary under `outputs/`. |

## Expected Data Layout

Each object should contain camera folders like this:

```text
data/sample/
  object_name/
    cam00/
      static.exr
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
- `outputs/train/inverse/`: inverse-stage checkpoint and predictions.
- `outputs/train/forward/`: forward-stage checkpoint and predictions.
- `outputs/eval_ictpolarreal_decomposition/`: CSV metrics and JSON summary.

## Flexible Usage

The default command is enough for the sample release. Use options only when
running on a different machine or dataset:

```bash
bash run.sh check-data
bash run.sh process --data-root /path/to/data --output-root /path/to/out
bash run.sh train --data-root /path/to/data --train-stage inverse --target albedo
bash run.sh train --data-root /path/to/data --train-stage forward --forward-input-mode gbuffer --forward-target static
bash run.sh evaluate --data-root /path/to/data --pred-root /path/to/predictions
```

Useful options:

- `--data-root PATH`: dataset location. Default: `data/sample`.
- `--output-root PATH`: output location. Default: `outputs`.
- `--torch-variant cpu`: force CPU PyTorch on machines without working CUDA.
- `--max-lights N`: use a sphere-wide subset for a quick diagnostic; the default
  346-light fit is recommended for material quality.
- `--backend torch --device cuda`: explicitly select the PyTorch optimizer.
- `--train-stage inverse|forward|both`: choose the training stage.
- `--input-mode polarization|gbuffer|image`: choose the inverse input representation.
- `--forward-input-mode gbuffer|polarization|image`: choose the forward input representation.
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
