# Preprocessing

Use the root release script to fit cross/parallel OLAT captures into material
g-buffer PNGs:

```bash
bash run.sh process
```

The implementation follows the original JAX decomposition. Cross-polarized
OLATs are fit with robust photometric stereo for diffuse albedo and normals.
The `parallel - cross` stack is fit with the anisotropic Ward model for
specular albedo, specular normals, sigma, roughness, anisotropy, tangent, and
bitangent. `auto` uses the PyTorch optimizer when CUDA is available and the
NumPy optimizer otherwise.

Polarization separation convention used internally:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`

Material maps are written under:

```text
outputs/material_acquisition/object_name/camXX/brdf/
```

The folder contains only material maps: `albedo.png`, `normal.png`, `specular.png`,
`roughness.png`, `sigma.png`, `anisotropy.png`, `tangent.png`, and
`bitangent.png`. Vector maps use the standard `[-1, 1]` to `[0, 1]` PNG
encoding; sigma and roughness use `x / (1 + x)` to retain high values.

The repository includes LSX light and camera calibration under `metadata/`.
Raw 350-frame sequences automatically skip indicator frames `0`, `1`, `348`,
and `349`. For a faster diagnostic, use `--max-lights 32`; publishable material
maps should use the default 346 lights.

For a custom location or explicit GPU execution:

```bash
bash run.sh process --data-root /path/to/data --output-root /path/to/output --backend torch --device cuda
```

On a Slurm cluster, submit the same acquisition as a one-GPU batch job so the
fit does not run on the login node:

```bash
bash run.sh process \
  --slurm \
  --env-name ictpolarreal \
  --backend torch \
  --device cuda \
  --slurm-account ACCOUNT \
  --slurm-partition PARTITION
```

The submit host validates the dataset before requesting a GPU, and the worker
validates it again after leaving the queue. By default the job requests one GPU,
16 CPU cores, 128 GB of host memory, and 3:59 hours. Slurm stdout/stderr logs go
under `outputs/slurm/`, while material maps keep the normal
`outputs/material_acquisition/` layout. Use `--slurm-dry-run` to inspect the
exact `sbatch` command without submitting it.
