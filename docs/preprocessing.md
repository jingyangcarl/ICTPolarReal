# Preprocessing

Use the release script to decompose cross/parallel OLAT captures into material
g-buffer PNGs. The implementation follows the original `external/gradient`
pipeline: cross-polarized OLATs estimate diffuse albedo/normal, and
`parallel - cross` estimates specular albedo, specular normal, sigma,
roughness, anisotropy, tangent, and bitangent. The default `auto` backend uses
PyTorch on GPU when available and falls back to NumPy on CPU.

```bash
bash scripts/ictpolarreal.sh process \
  --data-root /path/to/data \
  --output-root outputs \
  --max-lights 16 \
  --backend torch \
  --device cuda
```

Polarization separation convention used internally:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`

Material maps are written under:

```text
outputs/materials/object_name/camXX/material_properties/
```

Important files include `albedo.png`, `normal.png`, `specular.png`,
`roughness.png`, `sigma.png`, `anisotropy.png`, `tangent.png`, and
`bitangent.png`. If `LSX3_light_positions.txt` and
`LSX3_light_z_spiral.txt` are not found in the data/calibration folders, the
script falls back to deterministic synthetic light directions for smoke tests.
