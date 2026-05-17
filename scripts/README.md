## CrystalFormer-2D Scripts

Phase 1 supports the AWXL-to-layer-structure converter:

```bash
python scripts/awl2struct_2d.py samples.csv generated_structures --deduplicate
```

The converter writes CIF files plus JSON sidecars that preserve `pbc=[True, True, False]`.

Legacy 3D space-group post-processing, relaxation, energy ranking, novelty analysis, and plotting scripts from upstream CrystalFormer are not part of CrystalFormer-2D phase 1.
