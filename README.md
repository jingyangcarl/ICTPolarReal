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
| 4 | Acquire material maps | Uses the default polarized Ward decomposition, or the optional end-to-end Disney BRDF fit, then writes material PNG maps. |
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

- `outputs/material_acquisition/`: the default Ward material maps under
  `<object>/<camera>/brdf/`.
- `outputs/material_acquisition_end2end/`: independent OLAT-, HDRI-, and
  mixed-lighting Disney fits. Each camera has two entry points:
  `material/overview.png` compares the acquired maps, while
  `evaluation/overview.png` compares every fit on shared OLAT and HDRI cases.
  Detailed results are lighting-first under `evaluation/{olat,hdri}/`, and
  `run.json` records the complete acquisition run.
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
bash run.sh process --slurm --backend torch --device cuda --slurm-account ACCOUNT --slurm-partition PARTITION
bash run.sh process --material-acquisition end2end --end2end-hdri-root /path/to/hdr_maps_1k --slurm --backend torch --device cuda
bash run.sh train --data-root /path/to/data --train-stage inverse
bash run.sh train --data-root /path/to/data --train-stage forward --forward-mode gbuffer
bash run.sh evaluate --data-root /path/to/data --pred-root /path/to/predictions
```

Useful options:

- `--data-root PATH`: dataset location. Default: `data/sample`.
- `--output-root PATH`: output location. Default: `outputs`.
- `--material-acquisition default|end2end`: keep the current polarized Ward
  decomposition (`default`, and the implicit choice when omitted) or run the
  differentiable Disney BRDF acquisition (`end2end`).
- `--imaginaire-root PATH`: path to the external Imaginaire checkout used by
  `end2end`. The default is the sibling folder `../imaginaire`.
- `--end2end-steps N` and `--end2end-learning-rate FLOAT`: control the Disney
  optimization. Defaults are 33,000 steps and a learning rate of `1e-3`.
- `--end2end-tv-kind l1|edge-charbonnier|impulse-median|frequency-consensus`:
  choose the scalar-map regularizer. The default remains `impulse-median`, which
  completes the ordinary data fit, freezes a conservative 5x5 median/MAD
  detector away from albedo and normal edges, then applies a post-fit proximal
  update only at isolated score peaks. The optional `frequency-consensus` mode
  also leaves the full data-only fit unchanged, then performs one deterministic
  frozen update guided by albedo, normal, and 3x3/7x7 scalar-map neighborhoods.
  `edge-charbonnier` and `l1` retain the broader spatial-TV ablations.
- `--end2end-tv-weight FLOAT`: set the regularizer strength. For
  `impulse-median`, cumulative constrained-map shrink is the weight multiplied
  by `round(0.1 * end2end_steps)`; the `1.25e-3` default intentionally resolves
  selected peaks to the `0.005` dead zone. Use a smaller value for partial
  shrink or `0` for an unregularized ablation. For the TV modes this remains the
  objective coefficient. Unflagged scalar entries, frozen albedo/normal maps,
  and object boundaries are not changed. Each enabled impulse fit records the
  exact frozen targets and masks in a hashed `impulse_median_frozen.npz`
  artifact. For `frequency-consensus`, the effective update strength is
  `min(weight / 0.00125, 1)`: `0.00125` is full reference strength and `0`
  preserves the data-only result. Its hashed sources, targets, masks, and guide
  state are written to `frequency_consensus_frozen.npz`.
- `--end2end-profiles LIST`: choose independent `olat`, `hdri`, and `mix` fits.
  The default is `olat,hdri,mix`; `all` is an alias, and a subset such as
  `--end2end-profiles olat` avoids fitting the other profiles.
- `--end2end-primary-profile olat|hdri|mix`: select which requested profile is
  exposed to downstream loaders by each camera's `manifest.json`. The default
  is `olat`.
- `--end2end-eval-lights N`: reserve up to `N` calibrated OLATs from fitting and
  use them only for end-to-end relighting evaluation. On a complete LSX capture,
  the SuperDimension-style LSX adaptation fits 164 evenly sampled visible
  indices from `0..172` and has nine omitted visible holdouts; the default 16
  requests all nine. Rear/unused indices `173..345` are excluded. At least four
  fit lights are retained for reduced or non-LSX inputs; `0` reports fitted-light
  reconstruction instead of a held-out result.
- `--end2end-hdri-root PATH`: folder of HDR/EXR environment maps. It must be
  visible on the worker; the repository default points at the Maxine cluster
  HDRI collection.
- `--end2end-hdri-count N`, `--end2end-eval-hdris N`, and
  `--end2end-hdri-rotations N`: control the natural HDRI fit identities,
  held-out identities, and yaw rotations. Defaults are 100, 4, and 4. Generated
  white/red/green/blue calibration environments are added to the fit suite.
- `--torch-variant cpu --device cpu`: use CPU for diagnostics; diffusion training is slow without CUDA.
- `--max-lights N`: use a sphere-wide subset for a quick diagnostic; keep the
  default 346-light input to reproduce the exact 164-fit/9-holdout selection.
- `--backend torch --device cuda`: explicitly select the PyTorch optimizer.
- `--slurm`: submit material acquisition to Slurm instead of running it in the
  current shell. Resource options include `--slurm-account`, `--slurm-partition`,
  `--slurm-time`, `--slurm-cpus`, `--slurm-mem`, and `--slurm-gpus`.
- End-to-end acquisition is guarded against accidental execution on a local or
  login shell: use `--slurm`, or run the printed command only inside an
  allocation. The default Ward acquisition remains available locally.
- HDRI/mix fitting preflights the allocated device and requires at least a
  40 GiB GPU. The default three-profile run performs 99,000 total updates; if a
  queue time limit interrupts it, submit the identical command again to skip
  completed profiles and resume the active checkpoint.
- `--slurm-dry-run`: validate data and print the fully escaped `sbatch` command
  without submitting a job. Logs default to `outputs/slurm/`.
- `--train-stage inverse|forward|both`: choose the training stage.
- `--inverse-workflow pbr|polarization|both`: choose inverse supervision targets.
- `--forward-mode gbuffer|polarization|both`: choose the forward conditioning representation.
- `--train-steps N`: set optimizer steps for each selected model; the default 20-step run is a pipeline check.
- `--train-dry-run`: validate all tensors without loading diffusion checkpoints.
- `--resume latest`: continue from the newest checkpoint in each selected stage.
- `--train-eval-steps N`: periodically compare frozen pretrained and current fine-tuned weights.
- `--train-eval-samples N`: set the fixed evaluation subset size; `0` disables in-training evaluation.
- `--material-root PATH`: override the mode-specific material output root or use
  precomputed material maps from another run.
- `--skip-setup`: reuse the current environment.

The end-to-end mode does not vendor Imaginaire. Its source code, license, and
runtime dependencies remain external to this repository. Obtain an authorized
Imaginaire checkout, comply with its license, and install PyTorch, torchvision,
SciPy, NumPy, and Pillow in the environment used by the Slurm worker. The two
acquisition modes write to separate roots automatically: `default` uses
`outputs/material_acquisition`, while `end2end` uses
`outputs/material_acquisition_end2end`. An explicit `--material-root` overrides
the selected root.

ICTPolarReal end-to-end acquisition uses the measured parallel-polarized OLAT
image as its target and initializes frozen base color from the dataset
`albedo.exr`, together with the photometric normal and a constant optical-axis
view. Pixels failing the `n dot v > 0` validity gate are excluded. By default,
the data-only fit completes before the impulse detector freezes local
median/MAD targets inside that fitting mask and away from measured albedo or
normal edges. The post-fit proximal step updates only isolated score peaks;
all unflagged scalar entries remain bit-identical to the data fit. The broader
edge-aware Charbonnier and uniform L1 TV modes remain explicit ablations.
Invalid/background pixels are masked before whole-image 99.5th-percentile
linear scaling.

With `--end2end-tv-kind frequency-consensus`, the same full data-only fit is
followed by one deterministic guide-aware frozen update. Completion also
requires same-checkpoint pre/post-cleanup mean MSE not to regress on either the
OLAT or HDRI evaluation suite. This guard is a minimum safety check, not proof
that texture or material quality improved; inspect both the material maps and
relighting evaluation. See [docs/evaluation.md](docs/evaluation.md) for the
controlled comparison report.

End-to-end HDRI targets are not separately photographed environment-light
captures. They are synthesized from the measured ICTPolarReal parallel OLAT
stack by projecting each environment map onto the calibrated light basis with
spherical-Voronoi solid-angle weights. Fit and evaluation Voronoi cells are
recomputed on their respective light bases, keeping both composites full-sphere
while preserving the strict split. Natural environment maps are ranked by
sampled-light variance. The requested top set is divided by HDRI identity, so
all rotations of each held-out identity stay out of fitting; `w/r/g/b`
calibration environments and their rotations are fit-only conditions.

The `olat`, `hdri`, and `mix` models are separate fits. Mixed fitting alternates
one block of HDRI rotations with one block of OLAT conditions. Every fitted
profile is then evaluated on exactly the same OLAT and HDRI suites, including
the deterministic OLAT holdout and held-out natural HDRI identities. A camera
`manifest.json` records all profiles and `primary_material_dir`; downstream
training follows that field instead of assuming a `brdf/` directory. See
`material/overview.png` for the map comparison and `evaluation/overview.png`
for the aligned relighting comparison. `evaluation/metrics.csv` and
`evaluation/summary.json` are their machine-readable evaluation companions;
`run.json` at the material root records run status and settings. Shared
lighting provenance is limited to `evaluation/assets/conditions.json` and
`evaluation/assets/weights.npz`; each HDRI case carries the one lighting
thumbnail used by all profile predictions. The reported validity-masked MSE,
MAE, PSNR, and `ssim_global` compare clipped renderer-normalized LDR images.
They support within-run evaluation and do not establish numerical parity with
another dataset or pipeline.

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
