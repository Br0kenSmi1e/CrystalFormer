# CrystalFormer-2D

CrystalFormer-2D is a layer-group-specific fork of CrystalFormer for fixed layer group generation of 2D crystals.

The model treats fractional `x` and `y` as periodic coordinates and `z` as a non-periodic coordinate; generated structures use `pbc=[True, True, False]`.

Phase 1 supports:

```text
2D/layer dataset
-> layer-group preprocessing
-> train P(W, A, X, Y, Z, L | layergroup)
-> sample with --layergroup
-> convert sampled AWXL rows to layer CIF structures
```

Sampling requires a fixed layer group:

```bash
python main.py --optimizer none --restore_path RESTORE_PATH --layergroup 66 --num_samples 1000
```

Convert sampled rows:

```bash
crystalformer-2d-awl2struct samples.csv generated_structures --deduplicate
```

CrystalFormer-2D phase 1 does not include 3D space-group generation, formula conditioning, learned layergroup prediction, top-K group sampling, PPO, DPO, MCP, relaxation, energy ranking, novelty analysis, or plotting.
