# MicroAST

Source: https://github.com/EndyWon/MicroAST (AAAI 2023 official implementation).
Code and weights downloaded 2026-10-14. License: MIT, see LICENSE.txt.
The only change to net_microAST.py is package-relative function imports.
Inference uses CPU and torch.load(weights_only=True). No runtime downloads.
Released weights produced unstable values with partial feature modulation in
this environment; inference uses alpha=1 and intensity is blended in image space.
