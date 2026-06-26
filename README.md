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

Download the sample data from Google Drive and place it here:

```text
ICTPolarReal/
  data/
    sample/
      dragondruit/
        cam00/
```

Then run:

```bash
bash run.sh all
```

The script creates the environment, checks the data, prepares material previews,
runs a tiny training job, and evaluates the output. No manual Python setup is
needed for the default path.

## Expected Data Layout

Each object should contain camera folders like this:

```text
data/sample/
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

The sample Drive folder already follows this layout. If Drive download fails
from the command line, download it in a browser and keep the same folder
structure under `data/sample/`.

## Outputs

Default outputs are written to `outputs/`:

- `outputs/materials/`: diffuse/specular previews from OLAT polarization pairs.
- `outputs/train_albedo/`: checkpoint and prediction images from the baseline.
- `outputs/eval_ictpolarreal_decomposition/`: CSV metrics and JSON summary.

## Flexible Usage

The default command is enough for the sample release. Use options only when
running on a different machine or dataset:

```bash
bash run.sh check-data
bash run.sh process --data-root /path/to/data --output-root /path/to/out
bash run.sh train --data-root /path/to/data --target albedo
bash run.sh evaluate --data-root /path/to/data --pred-root /path/to/predictions
```

Useful options:

- `--data-root PATH`: dataset location. Default: `data/sample`.
- `--output-root PATH`: output location. Default: `outputs`.
- `--torch-variant cpu`: force CPU PyTorch on machines without working CUDA.
- `--max-lights N`: limit OLAT lights for a quick check.
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
