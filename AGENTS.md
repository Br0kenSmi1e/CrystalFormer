# Agent Instructions

Read [README.md](README.md) first. It is the source of truth for using the CrystalFormer-2D CLI.

This repository should be used through the CLI unless the user is explicitly asking for debugging or code changes.

If the user asks you to sample structures and does not provide a checkpoint, ask:

```text
Which CrystalFormer-2D checkpoint should I use for --restore_path?
```

Do not guess checkpoint paths or use local experiment checkpoints for user-facing sampling unless the user explicitly chooses one.
