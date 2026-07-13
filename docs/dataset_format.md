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

With `--material-acquisition end2end`, each camera has one material subtree and
one evaluation subtree. Lighting profiles are rows inside those two functions,
rather than top-level model directories:

```text
outputs/material_acquisition_end2end/
  run.json
  object_name/camXX/
    manifest.json
    material/
      overview.png
      olat/
        acquisition.json
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
      hdri/...
      mix/...
    evaluation/
      overview.png
      metrics.csv
      summary.json
      olat/
        comparison.png
        cases/<frame_id>/
          reference.png
          predictions/{olat,hdri,mix}.png
          errors/{olat,hdri,mix}.png
          comparison.png
      hdri/
        comparison.png
        cases/<condition_id>/
          lighting.png
          reference.png
          predictions/{olat,hdri,mix}.png
          errors/{olat,hdri,mix}.png
          comparison.png
      assets/
        conditions.json
        weights.npz
```

The default `--end2end-profiles olat,hdri,mix` creates all three profile
subtrees as independent fits. A subset creates only the requested directories.
The camera `manifest.json` lists the available profiles and contains
`primary_material_dir`, for example `material/olat/maps`. The RGB2X loaders
resolve this field first, so `--end2end-primary-profile` selects the maps
exposed downstream without copying or flattening files. Legacy roots without a
manifest still use the `brdf/`, `material_properties/`, and camera-root
fallbacks.

`evaluation/assets/conditions.json` records the ranked natural HDRI
identities, strict fit/held-out identity split, rotations, generated `w/r/g/b`
calibration conditions, and source hashes. `weights.npz` contains their
spherical-Voronoi weights on the calibrated ICT light basis. The lighting
assets directory intentionally contains only these two provenance files; the
HDRI thumbnail used by a report is stored with that evaluation case. HDRI
targets are weighted combinations of measured parallel-polarized OLAT images,
not independently captured environment-lit frames.

For a complete LSX capture, end-to-end acquisition uses a SuperDimension-style
164-light selection adapted to the ICTPolarReal rig: it considers visible
calibrated light indices `0..172`, fits 164 evenly sampled indices, and uses the
nine omitted visible indices as the OLAT evaluation set. Indices `173..345` are
rear or unused for this flow and are excluded rather than normalized into noisy
targets. The default `--end2end-eval-lights 16` therefore requests all nine
available visible holdouts. It also has 100 natural HDRI fit identities and
four held-out identities, each with four rotations; all rotations of one HDRI
identity remain in the same split. Every requested profile is evaluated on the
same OLAT and HDRI suites. OLAT case folders retain original capture frame IDs,
while HDRI case folders use the condition IDs recorded in
`evaluation/assets/conditions.json`.

`material/overview.png` is a profile-by-map grid for base color, normal,
roughness, and specular. `evaluation/overview.png` aligns one representative
OLAT and HDRI case across all profile predictions; suite-level
`comparison.png` files provide the larger lighting-specific views.
`evaluation/summary.json` describes the shared cases and
`evaluation/metrics.csv` contains one aggregate row per training profile and
evaluation lighting type. With all default profiles this is a six-row table.
References and lighting thumbnails are stored once per case, while predictions
and errors are keyed by training profile. Invalid/background pixels are masked
before the same renderer-native whole-image 99.5th-percentile linear scaling
and clipping used by the Imaginaire flow. The images do not preserve absolute
radiometric HDR scale and should not be treated as evidence of numerical
parity with another dataset.

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
