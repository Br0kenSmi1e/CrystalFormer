# Agent Instructions

This repository should be used through the CrystalFormer-2D CLI unless the user is explicitly asking for debugging or code changes. Read this file before inspecting the codebase.

## Purpose

CrystalFormer-2D is a layer-group-only CLI tool for fixed-layergroup 2D crystal generation.

The supported user workflow is:

```text
trained checkpoint
-> sample with python main.py --optimizer none --layergroup ...
-> convert samples.csv with crystalformer-2d-awl2struct
```

## Read First

For user-facing CLI usage, read:

- [docs/CLI_QUICKSTART.md](docs/CLI_QUICKSTART.md)
- [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)

Do not inspect model internals unless a command fails or the user asks for code-level work.

## Sampling Policy

If the user asks you to sample structures and does not provide a checkpoint, ask:

```text
Which CrystalFormer-2D checkpoint should I use for --restore_path?
```

No default released CrystalFormer-2D checkpoint is currently bundled.

Do not guess checkpoint paths. Do not use test-only or local experiment checkpoints for user-facing sampling unless the user explicitly chooses one.

## Required Sampling Inputs

Sampling requires:

- `--restore_path`: trained checkpoint directory or checkpoint file.
- `--layergroup`: fixed layer group, integer `1` to `80`.
- `--num_samples`: number of rows to sample.
- `--save_path`: output directory for `samples.csv`.

## Core Commands

Install:

```bash
pip install -e .
```

If the active shell does not expose `python`, use the environment runner instead:

```bash
uv run python main.py --help
```

Sample:

```bash
python main.py --optimizer none \
  --restore_path PATH_TO_CHECKPOINT \
  --layergroup 66 \
  --num_samples 100 \
  --save_path runs/lg66
```

Convert sampled rows:

```bash
crystalformer-2d-awl2struct \
  runs/lg66/samples.csv \
  runs/lg66/structures
```

## Invariants

- Use `layergroup`, not `spacegroup`, in public usage.
- Layer groups are integers from `1` to `80`.
- Sampling uses one fixed `--layergroup`; it does not sample layer groups.
- Generated structures use `pbc=[True, True, False]`.
- Fractional `x` and `y` are periodic coordinates.
- Fractional `z` is non-periodic.

## Unsupported User-Facing Features

These features are not part of the current CrystalFormer-2D CLI workflow:

- 3D space-group generation.
- Formula conditioning.
- Learned group prediction.
- Top-K group sampling.
- PPO, DPO, MCP, and reward workflows.
- Relaxation, energy ranking, novelty analysis, and plotting pipelines.

If the user asks for one of these features, explain that it is not currently supported by the CrystalFormer-2D phase-1 CLI.
