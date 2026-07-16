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

HDRI reference images are synthesized from measured ICTPolarReal
parallel-polarized OLATs using spherical-Voronoi solid-angle weights. They are
not separately captured environment-lit photographs. For the held-out HDRI
suite, synthesis and rendering use the nine held-out visible OLATs as well as
held-out environment identities. Voronoi cells are recomputed on the held-out
direction basis rather than dropping cells assigned to fit lights, so the
coarse held-out composite still covers the full sphere. The lighting and camera
manifests record this target origin.

Evaluation is organized by test lighting, not by the profile used for fitting:

```text
outputs/material_acquisition_end2end/<object>/<camera>/
  material/
    overview.png
    <profile>/maps/...
  evaluation/
    overview.png
    metrics.csv
    summary.json
    olat/
      comparison.png
      cases/<frame_id>/
        reference.png
        predictions/<profile>.png
        errors/<profile>.png
        comparison.png
    hdri/
      comparison.png
      cases/<condition_id>/
        lighting.png
        reference.png
        predictions/<profile>.png
        errors/<profile>.png
        comparison.png
    assets/
      conditions.json
      weights.npz
```

`evaluation/metrics.csv` has one aggregate row for every training profile and
evaluation lighting pair. The default `olat`, `hdri`, and `mix` fits therefore
produce six rows. Rows include the split, count, MSE, MAE, PSNR,
`ssim_global`, appearance diagnostics, and the profile and lighting labels.
`evaluation/summary.json` records the same matrix, the representative case
selection, target origin, and artifact paths.

OLAT case names retain original capture frame IDs. HDRI case names use the
condition IDs from the lighting manifest. A case stores its measured or
synthesized reference once, then keeps each fit's render and error under
`predictions/<profile>.png` and `errors/<profile>.png`. HDRI cases additionally
store one shared `lighting.png`. Each case's `comparison.png` provides a
readable side-by-side detail; the larger `comparison.png` at each suite root
collects those cases without repeating source files in the directory tree.
Error images use mean absolute RGB error on one fixed `0.00` to `0.25` scale,
so their colors are comparable across profiles and cases rather than being
independently stretched.

Shared lighting provenance is under `evaluation/assets/conditions.json` and
`evaluation/assets/weights.npz`. There is no separate preview dump; each HDRI
case stores only the lighting thumbnail needed by that case and the report.
The aggregate appearance diagnostics are `gt_mean_intensity`,
`pred_mean_intensity`, `mean_intensity_ratio`, and `luminance_correlation`;
they make a systematically dark or structurally mismatched reconstruction
obvious even when aggregate exposure-normalized metrics look less severe.

At camera level, `material/overview.png` is a compact profile-by-map grid for
base color, normal, roughness, and specular. `evaluation/overview.png` is the
primary relighting dashboard: it presents the aggregate metric matrix and the
same representative OLAT and HDRI cases for every fit. Detailed errors remain
in the suite comparisons rather than crowding the main dashboard.
`manifest.json` links both overviews, the root evaluation tables, and the
profile acquisitions. There is no `evaluation/report/` directory and no
training-profile-first `evaluation/<profile>/<lighting>/` hierarchy.

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

## Controlled regularization report

Compare a completed zero-weight camera result against its regularized result
with the acquisition-aware report composer. The two runs must use controlled
data, profile, optimization, lighting, and evaluation settings:

```bash
OBJECT=dragondruit
python -m ictpolarreal.processing.compare_regularization \
  --baseline outputs/material_regularizer_cam07/baseline/${OBJECT}/cam07 \
  --regularized outputs/material_regularizer_cam07/regularized/${OBJECT}/cam07 \
  --output outputs/material_regularizer_cam07/comparison \
  --data-root data/cam07_only
```

The report root contains `overview.png`, `summary.json`, `metrics.csv`, one
material sheet per fit profile under `material/`, and OLAT/HDRI relighting
sheets under `evaluation/`. A frequency-consensus comparison also includes
`material/frequency_hotspot_1to1.png` and
`material/frequency_fullmaps_1to1.png` for native-scale inspection.

Qualification requires exactly the `olat`, `hdri`, and `mix` fit profiles and
all numeric gates. A subset requested with `--profiles` is useful for an early
diagnostic, but its status is `INCOMPLETE` even when every available numeric
gate passes; only full three-profile coverage can report `PASS`. A passing
report still does not prove general texture preservation or overall material
quality. Inspect the native-resolution material sheets together with the
relighting comparisons and machine-readable metrics.
