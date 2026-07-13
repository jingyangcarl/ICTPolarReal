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

The default full capture contains 346 calibrated OLAT pairs. The
SuperDimension-style split adapted to LSX uses visible calibrated indices
`0..172`: 164 evenly sampled lights are fit and the nine omitted visible lights
are held out. Rear/unused indices `173..345` do not enter acquisition or
evaluation. Training OLAT targets and training HDRI composites use only the 164
fit observations. The nine omitted visible OLATs are used only in evaluation.
The default `--end2end-eval-lights 16` requests all nine available holdouts.

Natural HDRIs are ranked by their variance on the calibrated light basis and
split independently by source identity. The defaults retain 100 fit identities
and four held-out identities, with four yaw rotations each. Every rotation of a
held-out identity remains out of the HDRI and mixed fit pools. Generated
white/red/green/blue calibration environments and their rotations are fit-only.

HDRI target images are synthesized from measured ICTPolarReal
parallel-polarized OLATs using spherical-Voronoi solid-angle weights. They are
not separately captured environment-lit ground truth. For the held-out HDRI
suite, synthesis and rendering use the nine held-out visible OLATs as well as
held-out environment identities. Voronoi cells are recomputed on the held-out
direction basis rather than dropping cells assigned to fit lights, so the
coarse held-out composite still covers the full sphere. The lighting and camera
manifests record this target origin.

Each profile writes evaluation artifacts under:

```text
outputs/material_acquisition_end2end/<object>/<camera>/
  evaluation/<profile>/
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

The combined profile-level `evaluation/<profile>/metrics.csv` has one aggregate
OLAT row and one aggregate HDRI row. The nested tables retain per-case details:

- OLAT rows include split, stack index, calibrated light index, original frame
  ID, MSE, MAE, PSNR, `ssim_global`, appearance diagnostics, and artifact
  paths.
- HDRI rows include split, condition/source identity, rotation, the same
  metrics, and lighting/artifact paths.
- OLAT comparison panels contain ground truth, prediction, and 4x absolute
  error. HDRI panels add the environment-map preview.

Shared lighting provenance is under `evaluation/lighting/conditions.json` and
`evaluation/lighting/weights.npz`. There is no separate preview dump; each HDRI
case stores only the lighting thumbnail needed by that case and the reports.
The per-case appearance diagnostics are `gt_mean_intensity`,
`pred_mean_intensity`, `mean_intensity_ratio`, and `luminance_correlation`;
they make a systematically dark or structurally mismatched reconstruction
obvious even when aggregate exposure-normalized metrics look less severe.

At camera level, `evaluation/report/overview.png` presents all trained profiles
as rows using the same representative OLAT and HDRI cases. It also shows base
color, normal, roughness, specular, aggregate OLAT/HDRI metrics, predictions,
and errors. `evaluation/report/metrics.csv` and
`evaluation/report/summary.json` contain the aligned matrix, while
`manifest.json` links the report and profile acquisitions. Material maps live
separately under `material/<profile>/maps`.

These are validity-masked, scale-normalized LDR metrics. The capture mask is
intersected with the `n dot v > 0` front-facing gate; invalid pixels are zeroed
before the renderer-native whole-image 99.5th-percentile linear scaling and
clipping to `[0,1]`. This exposure normalization makes the evaluation useful
for spatial and reflectance-appearance agreement, but it removes absolute
intensity scale. Do not interpret the reported MSE, MAE, PSNR, or
`ssim_global` as radiometric HDR accuracy or as a claim of numerical parity
with another acquisition pipeline or dataset.

`ssim_global` is the repository's lightweight whole-foreground approximation,
not a windowed Gaussian SSIM implementation. Setting
`--end2end-eval-lights 0` or `--end2end-eval-hdris 0` changes the corresponding
suite to fitted-condition reconstruction, and the summary labels it
`fitted_olat` or `fitted_hdri` rather than held out.
