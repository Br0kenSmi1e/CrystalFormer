# CrystalFormer-2D Model Card

CrystalFormer-2D models layer-group-conditioned 2D crystal generation. The phase-1 probability model is:

```text
P(W_1, A_1, X_1, Y_1, Z_1, ..., W_n, A_n, X_n, Y_n, Z_n, L | layergroup)
```

Layer groups are represented internally as integer `G` values from 1 to 80. Public commands and documentation use `layergroup`.

The model treats fractional `x` and `y` as periodic coordinates and `z` as a non-periodic coordinate. Generated structures use `pbc=[True, True, False]`.
