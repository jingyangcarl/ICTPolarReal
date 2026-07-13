# Preprocessing

Material acquisition is one stage of the existing `run.sh` pipeline. The
default remains the polarized Ward decomposition:

```bash
bash run.sh process
```

The default path fits cross-polarized OLATs with robust photometric stereo for
diffuse albedo and normals. It fits the `parallel - cross` stack with the
anisotropic Ward model for specular albedo, specular normals, sigma, roughness,
anisotropy, tangent, and bitangent. `--backend auto` uses the PyTorch optimizer
when CUDA is available and the NumPy optimizer otherwise.

Select the Disney acquisition explicitly with `--material-acquisition
end2end`. It is CUDA-only and should be submitted as a one-GPU Slurm job on a
cluster:

```bash
bash run.sh process \
  --material-acquisition end2end \
  --imaginaire-root ../imaginaire \
  --end2end-hdri-root /path/to/hdr_maps_1k \
  --slurm \
  --backend torch \
  --device cuda
```

Omitting `--material-acquisition` is equivalent to
`--material-acquisition default`. The modes also use separate output roots:

- `default` writes `outputs/material_acquisition`.
- `end2end` writes `outputs/material_acquisition_end2end`.
- `--material-root PATH` overrides the root for the selected mode.

Both roots receive a `run.json`. It records the acquisition mode, settings,
camera list, timestamps, and `running`, `failed`, or `complete` status, so an
interrupted multi-camera run remains auditable.

## End-to-end lighting profiles

End-to-end acquisition fits three independent
`DisneyBRDFSimplifiedMultiLayer` models by default:

- `olat` fits measured, calibrated one-light-at-a-time targets.
- `hdri` fits environment-light targets synthesized from the measured OLATs.
- `mix` alternates one block of HDRI conditions and one block of OLAT
  conditions. The block size equals `--end2end-hdri-rotations`, which defaults
  to four.

These are separate material fits, not successive phases of one model. Select a
comma-separated subset with `--end2end-profiles`; `all` is an alias for the
default `olat,hdri,mix`:

```bash
# Fit only the OLAT profile and expose it downstream.
bash run.sh process \
  --material-acquisition end2end \
  --end2end-profiles olat \
  --end2end-primary-profile olat \
  --end2end-hdri-root /path/to/hdr_maps_1k \
  --slurm --backend torch --device cuda

# Fit all profiles but use the mixed maps in later training stages.
bash run.sh process \
  --material-acquisition end2end \
  --end2end-profiles all \
  --end2end-primary-profile mix \
  --end2end-hdri-root /path/to/hdr_maps_1k \
  --slurm --backend torch --device cuda
```

The primary profile must be included in the requested profile list. Each
camera's `manifest.json` records a `primary_material_dir` such as
`mix/simplified-multilayer/material/maps`. Downstream ICTPolarReal loaders read
that field, while preserving the legacy `brdf/` lookup for default Ward and
older material roots.

All requested profiles start from the same initialization. Base color and
normal are computed from the fitted OLAT subset using ICTPolarReal's polarized
decomposition, then frozen during each Disney fit. The adapter initializes the
Disney scalars in the renderer's sigmoid-logit storage space so their physical
starting values match the recorded configuration. Each profile has its own
optimizer, checkpoint, material state, provenance, and evaluation output.

End-to-end optimization defaults to 33,000 steps and a learning rate of
`1e-3`. Set these with `--end2end-steps` and
`--end2end-learning-rate`.

## HDRI conditions and synthesized targets

`--end2end-hdri-root` points to a folder of HDR, EXR, TIFF, or PNG latlong
environment maps. It is required even for an OLAT-only fit because every fitted
profile is evaluated on the common HDRI suite. The `run.sh` default refers to
the Maxine cluster HDRI collection; pass an explicit path elsewhere and make
sure the compute node can read it.

The environment preparation is deterministic:

1. Each natural environment is sampled at the camera's calibrated ICT light
   directions and ranked by sampled-light variance.
2. The top requested set is split by source identity into fit and held-out
   groups. All yaw rotations of one source stay in the same group.
3. Each latlong pixel is assigned to its nearest calibrated light, and
   spherical solid angle is integrated within the resulting Voronoi cell.
   The Voronoi assignment is recomputed separately on the fit-light and
   evaluation-light bases, so both composites cover the full sphere without
   borrowing directions from the other split.
4. The measured ICTPolarReal OLAT targets are combined with those RGB weights
   to synthesize the environment-lit target.

The defaults select 100 natural fit identities, four held-out natural
identities, and four yaw rotations per identity. Configure them with:

- `--end2end-hdri-count N`
- `--end2end-eval-hdris N`
- `--end2end-hdri-rotations N`

Generated white, red, green, and blue (`w/r/g/b`) calibration environments are
also added to the fit conditions, with the same rotations. They are never part
of the held-out natural-HDRI suite.

HDRI ground truth is therefore synthesized from measured ICTPolarReal OLATs;
it is not an independently captured HDRI-lit photograph. The camera manifest,
lighting condition manifest, and evaluation summaries record this target
origin. The output demonstrates that the same acquisition implementation can
be driven by OLAT, synthesized HDRI, or mixed targets, but it makes no claim of
numerical parity with SuperDimension or another dataset.

## Strict evaluation splits

The default full camera has 346 calibrated OLAT pairs. With
`--end2end-eval-lights 16`, the deterministic split reserves 16 measurements
before material initialization and optimization, leaving exactly 330 fit
lights. Training OLAT targets and training HDRI composites use only those fit
lights. The held-out OLATs are used only by evaluation, including as the
measured support for held-out-HDRI evaluation targets.

Natural HDRIs have a separate identity-level split controlled by
`--end2end-eval-hdris`. A held-out environment identity and all its rotations
are absent from the HDRI and mixed fit pools. Thus the default HDRI suite is
strictly held out both in environment identity and in its measured OLAT
support.

Every fitted profile is evaluated on the exact same OLAT cases and the exact
same HDRI identities/rotations. This produces an aligned profile-by-evaluation
matrix rather than a different test set for each model. Setting
`--end2end-eval-lights 0` changes the OLAT suite to sampled fitted-light
reconstruction. Setting `--end2end-eval-hdris 0` similarly uses fit HDRIs for
the HDRI evaluation suite; the summaries label these cases `fitted_olat` or
`fitted_hdri` rather than held out.

Targets use the polarization convention:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`
- `target = diffuse + specular`

Target and prediction are independently normalized over the foreground with
99.5th-percentile linear scaling and clipping. Metrics are foreground-masked.
They measure scale-normalized LDR appearance, not absolute radiometric HDR
accuracy.

## Output layout

Default Ward maps retain the legacy tree:

```text
outputs/material_acquisition/
  run.json
  object_name/camXX/brdf/
    albedo.png
    normal.png
    specular.png
    roughness.png
    sigma.png
    anisotropy.png
    tangent.png
    bitangent.png
```

Vector maps use the standard `[-1,1]` to `[0,1]` PNG encoding. Ward sigma and
roughness use `x / (1 + x)` so values above one are retained in the PNG.

End-to-end acquisition uses a profile-oriented camera tree:

```text
outputs/material_acquisition_end2end/
  run.json
  object_name/camXX/
    manifest.json
    lighting/
      conditions.json
      weights.npz
      previews/<condition_id>.png
    report/
      overview.png
      metrics.csv
      summary.json
    olat/simplified-multilayer/
      acquisition.json
      material/
        disney_brdf.pt
        maps/
          albedo.png
          baseColor.png
          normal.png
          specular.png
          roughness.png
          metallic.png
          specularTint.png
          subsurface.png
          anisotropic.png
          clearcoat.png
          clearcoatGloss.png
      evaluation/
        metrics.csv
        summary.json
        olat/
          metrics.csv
          summary.json
          contact_sheet.png
          cases/<frame_id>/{gt,pred,error,comparison}.png
        hdri/
          metrics.csv
          summary.json
          contact_sheet.png
          cases/<condition_id>/{lighting,gt,pred,error,comparison}.png
    hdri/simplified-multilayer/...
    mix/simplified-multilayer/...
```

Only requested profiles are created. `report/overview.png` uses one shared
representative OLAT case and one shared representative HDRI case across all
rows, alongside maps and aggregate metrics, so the profile comparison is
visually aligned. `report/metrics.csv` and `report/summary.json` provide the
same comparison in machine-readable form.

During each long fit, the resumable checkpoint is
`<profile>/simplified-multilayer/checkpoints/latest.pt`. Re-running with the
same inputs, profile, settings, material root, and Imaginaire source resumes
it. The checkpoint is removed after maps, model state, acquisition provenance,
and evaluations are written successfully. A later invocation validates those
completed artifacts and skips the matching profile. A mismatched signature
requires a different material root or explicit cleanup of the stale checkpoint.

## Capture selection and Slurm

The repository includes LSX light and camera calibration under `metadata/`.
Raw 350-frame sequences skip indicator frames 0, 1, 348, and 349. Use
`--max-lights N` for a diagnostic subset; the default 346-light selection is
recommended for a full acquisition. Reducing it also reduces the pool from
which the OLAT holdout is made, while always retaining at least four fit
lights.

Do not run the Disney optimizer on a GPU-less login node. Submit it through the
pipeline:

```bash
bash run.sh process \
  --material-acquisition end2end \
  --end2end-hdri-root /shared/path/to/hdr_maps_1k \
  --slurm \
  --backend torch \
  --device cuda \
  --slurm-account ACCOUNT \
  --slurm-partition PARTITION \
  --slurm-gpus 1
```

End-to-end acquisition requires exactly one GPU per job. HDRI and mixed fits
also require a GPU with at least 40 GiB memory; the worker checks the actual
device and estimated Disney autograd graph before allocating the fit. The
defaults request 16 CPU cores, 128 GB host memory, one GPU, and 3:59 hours. Use
`--slurm-time`, `--slurm-cpus`, `--slurm-mem`, and the other `--slurm-*`
options to match the cluster. `--slurm-dry-run` validates the inputs and prints
the escaped `sbatch` command without submitting it; logs default to
`outputs/slurm/`.

The default three profiles perform 33,000 updates each. On clusters with a
four-hour queue limit, a full run may need more than one submission. Re-run the
same command with the same material root: completed profiles are verified and
skipped, while the current profile resumes from `checkpoints/latest.pt`.

If setup runs on a GPU-less submit host, select a compatible CUDA PyTorch wheel
explicitly, for example `bash run.sh setup --torch-variant cu126`. The external
Imaginaire checkout and HDRI root must both be visible on the compute node.
Imaginaire is not bundled or relicensed here; users must obtain it, comply with
its license, and install its runtime dependencies in the worker environment.
