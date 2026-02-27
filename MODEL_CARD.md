# Model Card

We release two pretrained checkpoints, both trained on the [Alex-20s](https://huggingface.co/datasets/zdcao/alex-20s) dataset with the same model architecture but different `cfg_drop_prob` settings.

## CrystalFormer (Multi-task)

A unified model for both de novo generation (DNG) and crystal structure prediction (CSP), trained with `cfg_drop_prob=0.5`. The model seamlessly switches between DNG and CSP depending on whether a chemical formula is provided.

- **Weights**: [Google Drive](YOUR_LINK) | [Hugging Face](YOUR_LINK)

## CrystalFormer-CSP

A dedicated crystal structure prediction model, trained with `cfg_drop_prob=0` (formula conditioning is always enabled). This model is optimized for CSP tasks only.

- **Weights**: [Google Drive](https://drive.google.com/file/d/1sudBG-3AEm008_BiDE0y_m8AvNlVXzri/view?usp=sharing) | [Hugging Face](YOUR_LINK)

## Model Parameters

Both models share the same architecture:

```python
params, transformer = make_transformer(
        key=jax.random.PRNGKey(42),
        Nf=5,
        Kx=16,
        Kl=4,
        n_max=21,
        h0_size=256,
        num_layers=16,
        num_heads=8,
        key_size=32,
        model_size=256,
        embed_size=256,
        atom_types=119,
        wyck_types=28,
        dropout_rate=0.1,
        attn_dropout=0.1,
        widening_factor=4,
        sigmamin=1e-3
)
```

## Training Dataset

Alex-20s: contains ~1.7M general inorganic materials curated from the [Alexandria database](https://alexandria.icams.rub.de/), with $E_{hull} < 0.1$ eV/atom and no more than 20 Wyckoff sites in conventional cell. The dataset can be found in the [Hugging Face Datasets](https://huggingface.co/datasets/zdcao/alex-20s).

## Speeds, Sizes, Times
- Both models contain ~13.8 M parameters
- It takes 1058 seconds to generate a batch size 29,000 crystal samples on a single A100 GPU, which translates to a generation speed of 37 milliseconds per sample.