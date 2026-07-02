# CrystalFormer-2D

CrystalFormer-2D is a layer-group-only CLI tool for fixed-layergroup 2D crystal generation.

This README is the user and agent entry point. It explains the model context, the supported CLI workflow, checkpoint requirements, and common failure modes without requiring a codebase tour.

## Contents

- [What This Is](#what-this-is)
- [2D Crystal Representation](#2d-crystal-representation)
- [Model](#model)
- [Current CLI Status](#current-cli-status)
- [Install](#install)
- [Sample Structures](#sample-structures)
- [Convert Samples To Structures](#convert-samples-to-structures)
- [Train A Checkpoint](#train-a-checkpoint)
- [Checkpoints](#checkpoints)
- [Sampling Knobs](#sampling-knobs)
- [Common Errors](#common-errors)
- [Unsupported Features](#unsupported-features)
- [Relationship To Original CrystalFormer](#relationship-to-original-crystalformer)

## What This Is

CrystalFormer-2D models 2D crystal structures conditioned on a fixed layer group. The phase-1 workflow is:

```text
2D/layer dataset
-> train a layer-group-conditioned model
-> sample with a fixed --layergroup
-> convert sampled AWXL rows to layer CIF structures
```

Sampling does not choose a layer group automatically. The user specifies one layer group with `--layergroup`.

## 2D Crystal Representation

2D crystals are periodic in the in-plane `x` and `y` directions and non-periodic in the out-of-plane `z` direction. CrystalFormer-2D represents generated structures with `pbc=[True, True, False]`.

Instead of 230 3D space groups, 2D/layer structures are classified by 80 layer groups. Public commands and documentation use `layergroup`; the integer column name `G` is kept in sampled CSV files for compact compatibility with the model tensors.

A generated structure is represented by:

```text
G: layer group number, 1 to 80
W: Wyckoff indices for symmetry-inequivalent sites
A: atomic numbers
X: fractional coordinates [x, y, z] for symmetry-inequivalent sites
L: lattice parameters [a, b, c, alpha, beta, gamma]
M: active-site mask
```

The converter expands the symmetry-inequivalent sites using layer-group operations and writes structures with 2D periodic boundary conditions.

## Model

CrystalFormer-2D is an autoregressive transformer for:

```text
P(W_1, A_1, X_1, Y_1, Z_1, ..., W_n, A_n, X_n, Y_n, Z_n, L | layergroup)
```

The model samples Wyckoff positions and atom types as categorical variables. Fractional `x` and `y` are periodic variables; `z` is non-periodic. Lattice parameters are sampled after the atom sequence ends.

The atom sequence follows Wyckoff order, from higher-symmetry positions toward lower-symmetry positions. Padding atoms mark inactive sites; lattice parameters are sampled from the first padded position after active sites.

During training, variables fixed by the layer group and Wyckoff position are masked out of the loss.

## Current CLI Status

The default CrystalFormer-2D checkpoint is bundled under `checkpoints/default`.

Sampling requires a trained checkpoint directory or checkpoint file passed with `--restore_path`. Use `checkpoints/default` unless the user explicitly provides another checkpoint. If a user asks an agent to sample structures with a custom checkpoint but does not provide a path, the agent should ask:

```text
Which CrystalFormer-2D checkpoint should I use for --restore_path?
```

Agents should not guess checkpoint paths or use local experiment checkpoints for user-facing sampling unless the user explicitly chooses one.

## Install

From the repository root:

```bash
pip install -e .
```

This installs the Python package and the `crystalformer-2d-awl2struct` conversion command.

If your shell does not expose `python`, use your environment runner, for example:

```bash
uv run python main.py --help
```

## Sample Structures

Use `python main.py --optimizer none` for sampling:

```bash
python main.py --optimizer none \
  --restore_path checkpoints/default \
  --layergroup 66 \
  --num_samples 100 \
  --batchsize 64 \
  --save_path runs/lg66 \
  --Nf 5 \
  --Kx 16 \
  --Kl 4 \
  --n_max 21 \
  --atom_types 119 \
  --wyck_types 19 \
  --h0_size 256 \
  --num_layers 12 \
  --num_heads 8 \
  --key_size 32 \
  --model_size 64 \
  --embed_size 32 \
  --dropout_rate 0.4 \
  --attn_dropout 0.3
```

Required arguments:

- `--restore_path`: trained checkpoint directory or checkpoint file.
- `--layergroup`: fixed layer group in `[1, 80]`.

Common optional arguments:

- `--num_samples`: number of sampled rows to write.
- `--batchsize`: number of samples to generate per batch.
- `--save_path`: output directory.

The architecture flags in the example match `checkpoints/default/config.json`. Keep them consistent with the checkpoint being loaded.

The command writes:

```text
runs/lg66/samples.csv
```

The sampled CSV contains `G`, `L`, `X`, `A`, `W`, `M`, `logp_w`, `logp_xyz`, `logp_a`, and `logp_l`.

## Convert Samples To Structures

Convert sampled AWXL rows to layer structures:

```bash
crystalformer-2d-awl2struct \
  runs/lg66/samples.csv \
  runs/lg66/structures
```

To deduplicate converted structures:

```bash
crystalformer-2d-awl2struct \
  runs/lg66/samples.csv \
  runs/lg66/structures \
  --deduplicate
```

Expected conversion outputs include:

- `runs/lg66/structures/structures.csv`
- `runs/lg66/structures/structure_*.cif`
- `runs/lg66/structures/structure_*.json`

Generated structures should be treated as candidates. Stability, relaxation, energy ranking, and novelty analysis are outside the current phase-1 CLI.

## Train A Checkpoint

Training requires layer-group CSV datasets:

```bash
python main.py \
  --train_path train.csv \
  --valid_path val.csv \
  --save_path runs/train-lg \
  --run_name crystalformer-2d \
  --epochs 1000 \
  --val_interval 100
```

Training writes checkpoints under:

```text
runs/train-lg/<run_name>_<model_config>/epoch_XXXXXX.pkl
```

Use that run directory or a specific `epoch_XXXXXX.pkl` file as `--restore_path` for sampling.

## Checkpoints

The bundled default checkpoint uses this layout:

```text
checkpoints/default/
  epoch_000000.pkl
  config.json
```

`epoch_000000.pkl` is the model checkpoint.

`config.json` records the model flags needed to sample with the checkpoint, including values such as `Nf`, `Kx`, `Kl`, `n_max`, `h0_size`, `num_layers`, `num_heads`, `key_size`, `model_size`, `embed_size`, `dropout_rate`, and `attn_dropout`.

CrystalFormer-2D checkpoints are Python pickle files. Only load checkpoints from trusted sources.

## Sampling Knobs

`--temperature` controls the sampling distribution. Lower values make sampling sharper; higher values increase diversity.

`--top_p` controls nucleus sampling for categorical choices.

`--batchsize` controls how many samples are generated per batch.

`--num_samples` controls the total number of rows written to `samples.csv`.

Checkpoint architecture flags must match the checkpoint. If a checkpoint provides a `config.json`, use those values when sampling.

## Common Errors

`--restore_path is required when --optimizer none`

The sampling command is missing a checkpoint path.

`--layergroup is required when --optimizer none`

Sampling always requires a fixed layer group.

`--layergroup must be in [1, 80]`

CrystalFormer-2D uses layer groups, not 3D space groups.

`No checkpoint found at --restore_path ...`

The path does not exist, is not a checkpoint file, or does not contain `epoch_*.pkl` checkpoint files.

`Checkpoint ... is missing params`

The checkpoint file is not a CrystalFormer-2D model checkpoint.

Shape or parameter mismatch errors

The model flags used for sampling do not match the checkpoint architecture. Use the checkpoint's documented config.

## Unsupported Features

These features are not part of the current CrystalFormer-2D phase-1 CLI:

- 3D space-group generation.
- Formula conditioning.
- Learned group prediction.
- Top-K group sampling.
- Element-set constrained sampling through the CLI.
- MCMC sampling through the CLI.
- PPO, DPO, MCP, and reward workflows.
- Relaxation, energy ranking, novelty analysis, and plotting pipelines.

## Relationship To Original CrystalFormer

CrystalFormer-2D is a layer-group CLI branch. It is not the full upstream CrystalFormer 3D/CSP/RL feature set.

Use `layergroup` for public CrystalFormer-2D usage. Do not use upstream 3D `spacegroup` commands for this branch.
