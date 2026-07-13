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

## End-to-end material relighting evaluation

The optional Disney acquisition performs its own OLAT relighting evaluation as
part of `run.sh process`:

```bash
bash run.sh process \
  --material-acquisition end2end \
  --end2end-eval-lights 16 \
  --slurm \
  --backend torch \
  --device cuda
```

The default full capture contains 346 calibrated OLAT pairs. The split is made
before optimization: exactly 330 lights are fit and 16 deterministic lights
are held out. Held-out targets never contribute optimizer updates. Results for
each camera are written under
`outputs/material_acquisition_end2end/<object>/<camera>/brdf/`:

- `relighting/<frame_id>/gt.png`, `pred.png`, `error.png`, and
  `comparison.png` provide per-light evidence. The comparison panels are ground
  truth, prediction, and 4x absolute error.
- `relighting_metrics.csv` reports the split, stack index, calibrated light
  index, original frame ID, MSE, MAE, PSNR, and `ssim_global` for every held-out
  light.
- `relighting_summary.json` records aggregate metrics, normalization, and the
  exact held-out frame/light IDs.
- `relighting_contact_sheet.png` collects all held-out comparisons for visual
  inspection.

These are foreground-masked, scale-normalized LDR metrics. Each measured target
and rendered prediction is independently normalized using its 99.5th
percentile and clipped to `[0,1]` before comparison. Independent exposure
normalization makes the evaluation useful for spatial and reflectance-appearance
agreement, but it removes absolute intensity scale. Do not interpret the
reported MSE, MAE, PSNR, or `ssim_global` as radiometric HDR accuracy.

`ssim_global` is the repository's lightweight whole-foreground approximation,
not a windowed Gaussian SSIM implementation. This end-to-end evaluation is
generated directly by material acquisition and does not require a separate
`scripts/evaluate.sh` invocation.
