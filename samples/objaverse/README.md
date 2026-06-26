# Objaverse Evaluation Samples

The CVPR release supports Objaverse-style rendered samples in addition to real
ICTPolarReal captures. The evaluator expects rendered objects under:

```text
DATA_ROOT/
  renderings/
    object_name/
      lighting/
        all_white/frame_0001.png
        city/frame_0001.png
        olat_0/frame_0001.png
      gbuffers/
        albedo/Image0001.exr
        normal/Image0001.exr
        specular/Image0001.exr
        mask/Image0001.exr
```

Use `configs/eval_objaverse_samples.json` to name the sample objects and lights
available in your local copy. The original research workspace used samples such
as `table_vase`, `autumn_house`, and `brass_switch`; the public Google Drive
sample folder can add or replace these as uploaded.

Run decomposition evaluation:

```bash
bash scripts/evaluate.sh \
  --eval-mode objaverse \
  --eval-task decomposition \
  --data-root /path/to/objaverse_sample \
  --pred-root /path/to/predictions \
  --target albedo \
  --eval-manifest configs/eval_objaverse_samples.json
```

Run relighting evaluation:

```bash
bash scripts/evaluate.sh \
  --eval-mode objaverse \
  --eval-task relighting \
  --data-root /path/to/objaverse_sample \
  --pred-root /path/to/predictions \
  --eval-manifest configs/eval_objaverse_samples.json
```

