# Preprocessing

Use the root release script to fit cross/parallel OLAT captures into material
g-buffer PNGs:

```bash
bash run.sh process
```

The default implementation follows the original JAX decomposition.
Cross-polarized OLATs are fit with robust photometric stereo for diffuse albedo
and normals. The `parallel - cross` stack is fit with the anisotropic Ward model
for specular albedo, specular normals, sigma, roughness, anisotropy, tangent,
and bitangent. `auto` uses the PyTorch optimizer when CUDA is available and the
NumPy optimizer otherwise.

The same pipeline can instead run the Imaginaire-style end-to-end acquisition,
which optimizes a differentiable Disney BRDF against the calibrated
ICTPolarReal OLAT observations:

```bash
bash run.sh process \
  --material-acquisition end2end \
  --slurm \
  --backend torch \
  --device cuda
```

`--material-acquisition default` is implicit when the option is omitted.
This initial integration adapts Imaginaire's direct `olat` fitting branch; it
does not run SuperDimension's separate HDRI-synthesis or mixed-lighting modes.
End-to-end acquisition defaults to 33,000 optimizer steps and a learning rate
of `1e-3`; change them with `--end2end-steps` and
`--end2end-learning-rate`. Use `--imaginaire-root` when the external
Imaginaire checkout is not in the default sibling folder `../imaginaire`.

Polarization separation convention used internally:

- `diffuse = 2 * cross`
- `specular = 2 * max(parallel - cross, 0)`

Material maps are written under:

```text
outputs/material_acquisition/object_name/camXX/brdf/
```

The default folder contains only material maps: `albedo.png`, `normal.png`,
`specular.png`, `roughness.png`, `sigma.png`, `anisotropy.png`, `tangent.png`,
and `bitangent.png`. Vector maps use the standard `[-1, 1]` to `[0, 1]` PNG
encoding; sigma and roughness use `x / (1 + x)` to retain high values. The
end-to-end mode keeps the common `albedo.png`, `normal.png`, `specular.png`,
and `roughness.png` contract used by downstream training. It also writes
`baseColor.png`, `metallic.png`, `specularTint.png`, `subsurface.png`,
`anisotropic.png`, `clearcoat.png`, and
`clearcoatGloss.png`, plus the fitted `disney_brdf.pt` state and an
`acquisition.json` provenance/metrics record.

During a long Disney fit, `end2end_checkpoint.pt` is updated periodically in
the same folder. Re-running the identical command and material root resumes
that checkpoint automatically. It is removed after the final maps, state, and
metrics have been written successfully; changing the lights, resolution,
optimizer settings, or Imaginaire source requires a different material root.

Output roots are separated automatically for a side-by-side comparison. The
default mode writes to `outputs/material_acquisition`, while end-to-end mode
writes to `outputs/material_acquisition_end2end`. An explicit
`--material-root` overrides the corresponding default when a custom comparison
layout is needed.

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
under `outputs/slurm/`, while material maps keep the selected mode's output
layout. Use `--slurm-dry-run` to inspect the exact `sbatch` command without
submitting it.

End-to-end acquisition requires exactly one GPU per job and rejects any other
`--slurm-gpus` value. Imaginaire
is an external dependency: its source and license are not bundled or relicensed
by ICTPolarReal. The checkout must be available on the compute node, and its
runtime dependencies must be installed in the same environment activated by
the Slurm job. The imported Disney implementation requires PyTorch,
torchvision, SciPy, NumPy, and Pillow. Users are responsible for obtaining
Imaginaire and complying with its license.
