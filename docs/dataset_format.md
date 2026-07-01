# Dataset Format

ICTPolarReal stores each object as a directory with one folder per camera view.
The release code expects camera folders named `cam00` through `cam07`.

Required files for most scripts:

```text
object_name/camXX/static.exr
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
outputs/materials/object_name/camXX/material_properties/albedo.png
outputs/materials/object_name/camXX/material_properties/normal.png
outputs/materials/object_name/camXX/material_properties/roughness.png
outputs/materials/object_name/camXX/material_properties/specular.png
```

The training loader can use raw `polarization` mode from `static` plus a paired
OLAT cross/parallel image, or `gbuffer` mode from the processed material folder.

PNG or JPG versions may be used for smoke tests. Full experiments should use
linear EXR data and keep all data outside Git.

The native capture contains 350 numbered frames. Frames `0`, `1`, `348`, and
`349` are capture indicators, not OLAT measurements. The loader detects this
layout and maps frames `2..347` to calibrated light indices `0..345`. A
normalized layout numbered `0..345` is also supported with
`--frame-layout normalized`.
