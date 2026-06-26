# Preprocessing

Use the release script to convert cross/parallel OLAT captures into diffuse and
specular components. The default `auto` backend uses PyTorch on GPU when
available and falls back to NumPy on CPU.

```bash
bash scripts/ictpolarreal.sh process \
  --data-root /path/to/data \
  --output-root outputs \
  --max-lights 16 \
  --backend torch \
  --device cuda
```

The processing convention is:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`

Use `--preview` for tone-mapped PNGs. Omit it to write EXR outputs.
The bash script writes PNG previews and mean diffuse/specular material-property
summaries under `outputs/materials` by default.
