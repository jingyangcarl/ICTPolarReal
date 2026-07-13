# Evaluation

Use `scripts/evaluate.sh` for image-space material decomposition and relighting
metrics. The script supports both ICTPolarReal camera folders and Objaverse-style
rendered samples.

```bash
bash scripts/evaluate.sh \
  --eval-mode ictpolarreal \
  --eval-task decomposition \
  --data-root /path/to/data \
  --pred-root outputs/train/inverse/predictions \
  --eval-target albedo
```

Objaverse-style samples use `configs/eval_objaverse_samples.json`:

```bash
bash scripts/evaluate.sh \
  --eval-mode objaverse \
  --eval-task relighting \
  --data-root /path/to/objaverse_sample \
  --pred-root /path/to/objaverse/predictions \
  --eval-manifest configs/eval_objaverse_samples.json
```

The evaluator reports MSE, MAE, PSNR, and a lightweight global SSIM under the
object mask when available. It writes one CSV with per-sample metrics and one
JSON summary. Sparse-view reconstruction evaluation should be run through
external reconstruction systems using exported images from this repository.

## End-to-end material evaluation

The optional Disney acquisition evaluates material profiles as part of
`run.sh process`; it does not require a separate `scripts/evaluate.sh` call:

```bash
bash run.sh process \
  --material-acquisition end2end \
  --end2end-profiles olat,hdri,mix \
  --end2end-eval-lights 16 \
  --end2end-hdri-root /path/to/hdr_maps_1k \
  --end2end-hdri-count 100 \
  --end2end-eval-hdris 4 \
  --end2end-hdri-rotations 4 \
  --slurm \
  --backend torch \
  --device cuda
```

By default, `olat`, `hdri`, and `mix` are three independently optimized models.
`--end2end-profiles` can select any subset, while
`--end2end-primary-profile` only chooses which requested maps downstream
loaders consume. It does not change the evaluation matrix. Every fitted profile
is evaluated on the exact same OLAT suite and the exact same HDRI suite, which
makes rows comparable within one camera run.

The default full capture contains 346 calibrated OLAT pairs. The split is made
before polarized initialization or Disney optimization: exactly 330 lights are
fit and 16 deterministic lights are held out. Training OLAT targets and
training HDRI composites use only the 330 fit observations. Held-out OLATs are
used only in evaluation.

Natural HDRIs are ranked by their variance on the calibrated light basis and
split independently by source identity. The defaults retain 100 fit identities
and four held-out identities, with four yaw rotations each. Every rotation of a
held-out identity remains out of the HDRI and mixed fit pools. Generated
white/red/green/blue calibration environments and their rotations are fit-only.

HDRI target images are synthesized from measured ICTPolarReal polarized OLATs
using spherical-Voronoi solid-angle weights. They are not separately captured
environment-lit ground truth. For the held-out HDRI suite, synthesis and
rendering use the held-out OLAT support as well as held-out environment
identities. Voronoi cells are recomputed on the held-out direction basis rather
than dropping cells assigned to fit lights, so the coarse held-out composite
still covers the full sphere. The lighting and camera manifests record this
target origin.

Each profile writes evaluation artifacts under:

```text
outputs/material_acquisition_end2end/<object>/<camera>/
  <profile>/simplified-multilayer/evaluation/
    metrics.csv
    summary.json
    olat/
      metrics.csv
      summary.json
      contact_sheet.png
      cases/<frame_id>/
        gt.png
        pred.png
        error.png
        comparison.png
    hdri/
      metrics.csv
      summary.json
      contact_sheet.png
      cases/<condition_id>/
        lighting.png
        gt.png
        pred.png
        error.png
        comparison.png
```

The combined profile-level `evaluation/metrics.csv` has one aggregate OLAT row
and one aggregate HDRI row. The nested tables retain per-case details:

- OLAT rows include split, stack index, calibrated light index, original frame
  ID, MSE, MAE, PSNR, `ssim_global`, and artifact paths.
- HDRI rows include split, condition/source identity, rotation, the same
  metrics, and lighting/artifact paths.
- OLAT comparison panels contain ground truth, prediction, and 4x absolute
  error. HDRI panels add the environment-map preview.

At camera level, `report/overview.png` presents all trained profiles as rows
using the same representative OLAT and HDRI cases. It also shows base color,
normal, roughness, specular, aggregate OLAT/HDRI metrics, predictions, and
errors. `report/metrics.csv` and `report/summary.json` contain the aligned
matrix, while `manifest.json` links the report and profile acquisitions.

These are foreground-masked, scale-normalized LDR metrics. Each measured target
and rendered prediction is independently normalized using its 99.5th
percentile and clipped to `[0,1]` before comparison. Independent exposure
normalization makes the evaluation useful for spatial and reflectance-appearance
agreement, but it removes absolute intensity scale. Do not interpret the
reported MSE, MAE, PSNR, or `ssim_global` as radiometric HDR accuracy or as a
claim of numerical parity with another acquisition pipeline or dataset.

`ssim_global` is the repository's lightweight whole-foreground approximation,
not a windowed Gaussian SSIM implementation. Setting
`--end2end-eval-lights 0` or `--end2end-eval-hdris 0` changes the corresponding
suite to fitted-condition reconstruction, and the summary labels it
`fitted_olat` or `fitted_hdri` rather than held out.
