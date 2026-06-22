# Dataset Format

ICTPolarReal stores each object as a directory with one folder per camera view.
The release code expects camera folders named `cam00` through `cam07`.

Required files for most scripts:

```text
object_name/camXX/static.exr
object_name/camXX/mask.png
object_name/camXX/cross/000001.exr
object_name/camXX/parallel/000001.exr
```

Optional material targets:

```text
object_name/camXX/albedo.exr
object_name/camXX/normal.exr
object_name/camXX/specular.exr
```

PNG or JPG versions may be used for smoke tests. Full experiments should use
linear EXR data and keep all data outside Git.

