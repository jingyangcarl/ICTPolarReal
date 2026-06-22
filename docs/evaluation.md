# Evaluation

Use the decomposition evaluator for image-space material predictions and the
relighting evaluator for predicted relit images.

```bash
python -m ictpolarreal.eval.decomposition \
  --pred-root outputs/predictions \
  --gt-root /path/to/data \
  --target albedo
```

The evaluator reports MSE, MAE, and PSNR under the object mask. Sparse-view
reconstruction evaluation should be run through external reconstruction systems
using exported images from this repository.

