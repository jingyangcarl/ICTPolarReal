# Third-Party Notices

This repository contains original release code for ICTPolarReal plus small
utility patterns adapted from the ObjectReal research workspace.

The RGB2X training data contract and fine-tuning procedure are adapted from the
project's Lotus integration (`https://github.com/jingyangcarl/lotus`), based on
Lotus under Apache-2.0. The `zheng95z/rgb-to-x` and `zheng95z/x-to-rgb`
checkpoints derive from Adobe Research RGB2X and remain subject to their model
and research-license terms.

The training recipes interoperate with PyTorch, Diffusers, Accelerate, PEFT,
Transformers, ImageIO, NumPy, Pillow, and PyYAML. Those projects retain their
own licenses.

When porting additional code from ObjectReal submodules, preserve upstream
license headers and add the upstream project, source URL, and license here.

The optional `--material-acquisition end2end` mode interoperates at runtime
with an external NVIDIA Imaginaire checkout. No Imaginaire source is copied or
redistributed in this repository. Users must obtain that checkout separately
and comply with its NVIDIA Source Code License and runtime dependency licenses.
