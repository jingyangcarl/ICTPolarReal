# Preprocessing

Use `ictpolarreal.processing.prepare_materials` to convert cross/parallel OLAT
captures into diffuse and specular components.

```bash
python -m ictpolarreal.processing.prepare_materials \
  --data-root /path/to/data \
  --out-root outputs/materials \
  --max-lights 16 \
  --preview
```

The processing convention is:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`

Use `--preview` for tone-mapped PNGs. Omit it to write EXR outputs.

