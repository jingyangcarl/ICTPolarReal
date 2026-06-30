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
missing or incomplete, the script downloads a minimal runnable subset from the
sample Google Drive folder. If Google Drive blocks command-line download, open
the sample link in a browser and place the folder under `data/sample`, then
rerun the same command.

## What `run.sh all` Does

| Step | Action | Result |
| --- | --- | --- |
| 1 | Set up the environment | Creates or reuses the `ictpolarreal` environment and installs the package. |
| 2 | Check Python packages | Verifies imports and reports PyTorch/CUDA availability. |
| 3 | Prepare sample data | Validates `data/sample`; if it is missing, downloads a minimal sample subset. |
| 4 | Decompose polarization data | Writes material g-buffer PNG maps. |
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
      cross/000000.exr
      parallel/000000.exr
```

The automatic sample path requires `static`, `mask`, the training target
(`albedo` by default), and at least one paired `cross`/`parallel` OLAT image.
The full dataset may also include more material annotations, cameras, and OLAT
lights. `run.sh process` can derive release g-buffers from the raw OLAT pairs.

## Outputs

Default outputs are written to `outputs/`:

- `outputs/materials/`: decomposed material PNG maps.
- `outputs/train_inverse_albedo/`: inverse-stage checkpoint and predictions.
- `outputs/train_forward_static/`: forward-stage checkpoint and predictions.
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
- `--max-lights N`: limit OLAT lights for a quick check.
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
