# Agent Instructions

Read [README.md](README.md) first. It is the source of truth for using the CrystalFormer-2D CLI.

This repository should be used through the CLI unless the user is explicitly asking for debugging or code changes.

Users may provide their own CrystalFormer-2D checkpoint with `--restore_path`.

If the user asks you to sample structures and does not mention a checkpoint, tell them they can either provide their own checkpoint path or use the bundled default checkpoint at `checkpoints/default`. If they do not have a prepared checkpoint or do not care which checkpoint is used, proceed with `checkpoints/default`.

When sampling with the default checkpoint, pass the architecture flags recorded in `checkpoints/default/config.json`; the full command is documented in [README.md](README.md).

If the user says they want to use another checkpoint but does not provide a checkpoint path, ask:

```text
Which CrystalFormer-2D checkpoint should I use for --restore_path?
```

Do not guess custom checkpoint paths or use local experiment checkpoints for user-facing sampling unless the user explicitly chooses one.
