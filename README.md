# CrystalFormer-2D

CrystalFormer-2D is a layer-group-only CLI tool for fixed-layergroup 2D crystal generation.

The phase-1 workflow is:

```text
2D/layer dataset
-> train a layer-group-conditioned model
-> sample with a fixed --layergroup
-> convert sampled AWXL rows to layer CIF structures
```

The model uses layer groups `1` to `80`. Generated structures use `pbc=[True, True, False]`; fractional `x` and `y` are periodic, while `z` is treated as non-periodic.

## Start Here

- User-facing CLI workflow: [docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md)
- Checkpoint policy and expected layout: [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)
- Agent usage notes: [AGENTS.md](AGENTS.md)

No released CrystalFormer-2D checkpoint is currently bundled. Sampling requires a trained checkpoint directory or checkpoint file passed with `--restore_path`.

## Minimal Commands

Install the package and console tools:

```bash
pip install -e .
```

Sample from a trained checkpoint:

```bash
python main.py --optimizer none \
  --restore_path PATH_TO_CHECKPOINT \
  --layergroup 66 \
  --num_samples 100 \
  --save_path runs/lg66
```

Convert sampled rows to structures:

```bash
crystalformer-2d-awl2struct \
  runs/lg66/samples.csv \
  runs/lg66/structures
```

## Scope

CrystalFormer-2D phase 1 supports fixed-layergroup training, sampling, and AWXL-to-layer-structure conversion.

It does not include 3D space-group generation, formula conditioning, learned layergroup prediction, top-K group sampling, PPO, DPO, MCP, relaxation, energy ranking, novelty analysis, or plotting.
