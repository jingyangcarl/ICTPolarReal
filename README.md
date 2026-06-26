# ICTPolarReal

Official code release scaffold for **ICTPolarReal: A Polarized Reflection and
Material Dataset of Real World Objects**.

ICTPolarReal is a CVPR 2026 dataset and benchmark for real-world polarized
reflectance. The capture system records 218 everyday objects with 8 viewpoints,
346 OLAT lights, and cross/parallel polarization, enabling diffuse/specular
separation, material decomposition, relighting, and sparse-view reconstruction
experiments.

- Project page: https://jingyangcarl.github.io/ICTPolarReal/
- Paper: https://arxiv.org/abs/2603.24912
- Sample data: linked from the project page

## Installation

The easiest path is the all-in-one script:

```bash
bash scripts/ictpolarreal.sh all \
  --data-root /path/to/ICTPolarReal_sample \
  --download-sample
```

This creates an environment when needed, checks imports, validates the data
layout, processes OLAT cross/parallel captures into diffuse/specular material
previews, runs a tiny training smoke test, and evaluates the produced
predictions. If the data root is missing, the script prints the sample Google
Drive folder and can try `gdown` with `--download-sample`. If Google Drive
rate-limits or blocks a file, open the folder in a browser, copy the sample
locally, and rerun the same command without changing the layout.

For manual installation:

```bash
git clone https://github.com/jingyangcarl/ICTPolarReal.git
cd ICTPolarReal
conda create -n ictpolarreal python=3.10 -y
conda activate ictpolarreal
pip install -e ".[dev,train]"
```

For CPU-only machines or broken CUDA drivers, force the CPU PyTorch wheel:

```bash
bash scripts/ictpolarreal.sh setup --torch-variant cpu
```

## Dataset Layout

Commands expect a configurable `--data-root` with object folders:

```text
DATA_ROOT/
  object_name/
    cam00/
      static.exr
      mask.png
      albedo.exr
      normal.exr
      specular.exr
      cross/000000.exr
      parallel/000000.exr
```

PNG inputs are also supported for quick checks. Full-resolution EXR data should
stay outside Git history.

## Quickstart

One command:

```bash
DATA_ROOT=/path/to/data bash scripts/ictpolarreal.sh all
```

Individual stages:

Inspect a local or sample dataset:

```bash
bash scripts/ictpolarreal.sh check-data --data-root /path/to/ICTPolarReal/sample
```

Prepare diffuse/specular images from cross/parallel OLAT captures:

```bash
bash scripts/ictpolarreal.sh process \
  --data-root /path/to/data \
  --output-root outputs \
  --backend torch \
  --device cuda
```

Run a tiny inverse decomposition training smoke test:

```bash
bash scripts/ictpolarreal.sh train \
  --data-root /path/to/data \
  --target albedo \
  --train-steps 20 \
  --batch-size 1
```

Evaluate predicted outputs:

```bash
bash scripts/evaluate.sh \
  --eval-mode ictpolarreal \
  --eval-task decomposition \
  --data-root /path/to/data \
  --pred-root outputs/inverse_debug/predictions \
  --target albedo
```

Objaverse-style sample evaluation is documented in
`samples/objaverse/README.md` and uses `configs/eval_objaverse_samples.json`.

```bash
bash scripts/evaluate.sh \
  --eval-mode objaverse \
  --eval-task relighting \
  --data-root /path/to/objaverse_sample \
  --pred-root outputs/predictions \
  --eval-manifest configs/eval_objaverse_samples.json
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
