# Evaluation

Use `scripts/evaluate.sh` for image-space material decomposition and relighting
metrics. The script supports both ICTPolarReal camera folders and Objaverse-style
rendered samples.

```bash
bash scripts/evaluate.sh \
  --eval-mode ictpolarreal \
  --eval-task decomposition \
  --data-root /path/to/data \
  --pred-root outputs/predictions \
  --target albedo
```

Objaverse-style samples use `configs/eval_objaverse_samples.json`:

```bash
bash scripts/evaluate.sh \
  --eval-mode objaverse \
  --eval-task relighting \
  --data-root /path/to/objaverse_sample \
  --pred-root outputs/predictions \
  --eval-manifest configs/eval_objaverse_samples.json
```

The evaluator reports MSE, MAE, PSNR, and a lightweight global SSIM under the
object mask when available. It writes one CSV with per-sample metrics and one
JSON summary. Sparse-view reconstruction evaluation should be run through
external reconstruction systems using exported images from this repository.
