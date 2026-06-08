# CLI Quickstart

This guide shows how to use CrystalFormer-2D from the command line without reading the model internals.

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

## Checkpoint Requirement

Sampling requires a trained CrystalFormer-2D checkpoint passed with `--restore_path`.

No released CrystalFormer-2D checkpoint is currently bundled. If you do not already have a trained checkpoint, train one first or obtain one from a trusted release source.

See [CHECKPOINTS.md](CHECKPOINTS.md) for checkpoint layout and safety notes.

## Sample Structures

Use `python main.py --optimizer none` for sampling:

```bash
python main.py --optimizer none \
  --restore_path PATH_TO_CHECKPOINT \
  --layergroup 66 \
  --num_samples 100 \
  --batchsize 64 \
  --save_path runs/lg66
```

Required arguments:

- `--restore_path`: trained checkpoint directory or checkpoint file.
- `--layergroup`: fixed layer group in `[1, 80]`.
- `--num_samples`: number of sampled rows to write.
- `--save_path`: output directory.

The command writes:

```text
runs/lg66/samples.csv
```

The sampled CSV contains `G`, `L`, `X`, `A`, `W`, `M`, and log probability columns.

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

The generated structures use `pbc=[True, True, False]`.

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
