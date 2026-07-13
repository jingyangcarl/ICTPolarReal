# Dataset Format

ICTPolarReal stores each object as a directory with one folder per camera view.
The release code expects camera folders named `cam00` through `cam07`.

Required files for most scripts:

```text
object_name/camXX/static.exr
object_name/camXX/static_cross.exr
object_name/camXX/static_parallel.exr
object_name/camXX/mask.png
object_name/camXX/cross/000002.exr ... 000347.exr
object_name/camXX/parallel/000002.exr ... 000347.exr
```

Optional material targets:

```text
object_name/camXX/albedo.exr
object_name/camXX/normal.exr
object_name/camXX/specular.exr
```

Processed material maps are written separately so raw data remains unchanged:

```text
outputs/material_acquisition/object_name/camXX/brdf/albedo.png
outputs/material_acquisition/object_name/camXX/brdf/normal.png
outputs/material_acquisition/object_name/camXX/brdf/roughness.png
outputs/material_acquisition/object_name/camXX/brdf/specular.png
```

This is the default Ward acquisition root. With
`--material-acquisition end2end`, the same object/camera layout and common map
names are written under `outputs/material_acquisition_end2end/`, alongside the
additional `baseColor`, `metallic`, `specularTint`, `subsurface`,
`anisotropic`, `clearcoat`, and `clearcoatGloss` maps. The folder
also records `disney_brdf.pt` and `acquisition.json`. An explicit
`--material-root` overrides either mode-specific root.

With the default `--end2end-eval-lights 16`, a full 346-light camera is split
deterministically into 330 fit lights and 16 held-out relighting lights. The
held-out predictions use the original capture frame IDs:

```text
outputs/material_acquisition_end2end/object_name/camXX/brdf/
  relighting_metrics.csv
  relighting_summary.json
  relighting_contact_sheet.png
  relighting/000002/gt.png
  relighting/000002/pred.png
  relighting/000002/error.png
  relighting/000002/comparison.png
```

Only frames selected for the deterministic holdout are present below
`relighting/`; `relighting_summary.json` records their frame IDs, calibrated
light indices, normalization, and aggregate metrics. The images are clipped
LDR visualizations. Prediction and target are independently scale-normalized,
and metrics are foreground-masked, so these files do not preserve or evaluate
absolute radiometric HDR scale.

The RGB2X training loader pairs the processed albedo, normal, and specular maps
with static and calibrated OLAT observations. Inverse training predicts PBR or
cross/parallel targets from RGB. Forward training conditions on either the PBR
maps or canonical `static_cross`/`static_parallel` images plus per-light
irradiance.

PNG or JPG versions may be used for smoke tests. Full experiments should use
linear EXR data and keep all data outside Git.

The native capture contains 350 numbered frames. Frames `0`, `1`, `348`, and
`349` are capture indicators, not OLAT measurements. The loader detects this
layout and maps frames `2..347` to calibrated light indices `0..345`. A
normalized layout numbered `0..345` is also supported with
`--frame-layout normalized`.
