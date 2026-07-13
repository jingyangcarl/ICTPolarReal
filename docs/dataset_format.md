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

Processed material maps are written separately so raw data remains unchanged.
The default Ward acquisition keeps the legacy `brdf/` layout:

```text
outputs/material_acquisition/object_name/camXX/brdf/albedo.png
outputs/material_acquisition/object_name/camXX/brdf/normal.png
outputs/material_acquisition/object_name/camXX/brdf/roughness.png
outputs/material_acquisition/object_name/camXX/brdf/specular.png
```

Its root also contains `run.json`, which records settings, processed cameras,
timestamps, and completion status.

With `--material-acquisition end2end`, each requested lighting profile has its
own clean subtree instead of sharing a `brdf/` directory:

```text
outputs/material_acquisition_end2end/
  run.json
  object_name/camXX/
    manifest.json
    lighting/
      conditions.json
      weights.npz
      previews/<condition_id>.png
    report/
      overview.png
      metrics.csv
      summary.json
    olat/simplified-multilayer/
      acquisition.json
      material/
        disney_brdf.pt
        maps/
          albedo.png
          baseColor.png
          normal.png
          specular.png
          roughness.png
          metallic.png
          specularTint.png
          subsurface.png
          anisotropic.png
          clearcoat.png
          clearcoatGloss.png
      evaluation/
        metrics.csv
        summary.json
        olat/
          metrics.csv
          summary.json
          contact_sheet.png
          cases/<frame_id>/{gt,pred,error,comparison}.png
        hdri/
          metrics.csv
          summary.json
          contact_sheet.png
          cases/<condition_id>/{lighting,gt,pred,error,comparison}.png
    hdri/simplified-multilayer/...
    mix/simplified-multilayer/...
```

The default `--end2end-profiles olat,hdri,mix` creates all three profile
subtrees as independent fits. A subset creates only the requested directories.
The camera `manifest.json` lists the available profiles and contains
`primary_material_dir`, for example
`olat/simplified-multilayer/material/maps`. The RGB2X loaders resolve this field
first, so `--end2end-primary-profile` selects the maps exposed downstream
without copying or flattening files. Legacy roots without a manifest still use
the `brdf/`, `material_properties/`, and camera-root fallbacks.

`lighting/conditions.json` records the ranked natural HDRI identities, strict
fit/held-out identity split, rotations, generated `w/r/g/b` calibration
conditions, source hashes, and preview paths. `weights.npz` contains their
spherical-Voronoi weights on the calibrated ICT light basis. HDRI targets are
weighted combinations of the measured polarized OLAT images, not independently
captured environment-lit frames.

With the defaults, a full 346-light camera has 330 OLAT fit measurements and 16
strictly held-out measurements. It also has 100 natural HDRI fit identities and
four held-out identities, each with four rotations; all rotations of one HDRI
identity remain in the same split. Every requested profile is evaluated on the
same OLAT and HDRI suites. OLAT case folders retain original capture frame IDs,
while HDRI case folders use the condition IDs recorded in
`lighting/conditions.json`.

`report/overview.png` aligns one representative OLAT and HDRI case across the
profile rows and shows selected material maps and aggregate metrics.
`report/summary.json` and `report/metrics.csv` are its machine-readable
companions. The images and metrics use independent foreground 99.5th-percentile
normalization and clipping. They do not preserve absolute radiometric HDR scale
and should not be treated as evidence of numerical parity with another
dataset.

An explicit `--material-root` overrides either mode-specific output root.

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
