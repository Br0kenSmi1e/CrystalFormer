# Checkpoints

## Current Status

No released CrystalFormer-2D checkpoint is currently bundled in this repository.

Sampling requires a trained checkpoint directory or checkpoint file passed with `--restore_path`.

## Agent Policy

If a user asks an agent to sample structures and does not provide a checkpoint, the agent should ask:

```text
Which CrystalFormer-2D checkpoint should I use for --restore_path?
```

Agents should not guess checkpoint paths or use local experiment checkpoints for user-facing sampling unless the user explicitly chooses one.

## Expected Checkpoint Layout

When a released or demo checkpoint is added, use this layout:

```text
checkpoints/<checkpoint-name>/
  epoch_XXXXXX.pkl
  config.json
  README.md
  SHA256SUMS
```

`epoch_XXXXXX.pkl` is the model checkpoint.

`config.json` records the model flags needed to sample with the checkpoint, including values such as `Nf`, `Kx`, `Kl`, `n_max`, `h0_size`, `num_layers`, `num_heads`, `key_size`, `model_size`, `embed_size`, `dropout_rate`, and `attn_dropout`.

`README.md` describes the training dataset, intended usage, recommended layer groups if applicable, and limitations.

`SHA256SUMS` records checksums for released checkpoint files.

## Safety

CrystalFormer-2D checkpoints are Python pickle files. Only load checkpoints from trusted sources.

Do not treat generated samples as validated stable materials. Sampling produces candidate structures that may need separate validation, relaxation, or downstream filtering outside the current phase-1 CLI.

## Future Default Checkpoint

When a default released checkpoint is selected:

- Add it under `checkpoints/<checkpoint-name>/` or document the external release location.
- Update this file with the checkpoint name, source, config, and checksum.
- Update [AGENTS.md](../AGENTS.md) so agents know whether they may use it when no checkpoint is specified.
- Update [CLI_QUICKSTART.md](CLI_QUICKSTART.md) with a concrete first-run sampling command.
