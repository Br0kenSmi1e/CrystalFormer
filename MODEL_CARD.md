# Model Card

We release two pretrained checkpoints, both trained on the [Alex-20s](https://huggingface.co/datasets/zdcao/alex-20s) dataset with the same model architecture but different `cfg_drop_prob` settings.

## Multitask

A unified model for both de novo generation (DNG) and crystal structure prediction (CSP), trained with  `0 < cfg_drop_prob < 1`. The model seamlessly switches between DNG and CSP depending on whether a chemical formula is provided.

- **Weights**: [Google Drive](https://drive.google.com/file/d/1qr-e0C2KrgPhnDOv4sn-FWI0r2pdQ0vk/view?usp=sharing) | [Hugging Face](https://huggingface.co/zdcao/CrystalFormer/resolve/main/alex20s/multi/epoch_044000.pkl)

## CSP-Only

A dedicated crystal structure prediction model, trained with `cfg_drop_prob=0` (formula conditioning is always enabled). This model is optimized for CSP tasks only.

- **Weights**: [Google Drive](https://drive.google.com/file/d/1sudBG-3AEm008_BiDE0y_m8AvNlVXzri/view?usp=sharing) | [Hugging Face](https://huggingface.co/zdcao/CrystalFormer/resolve/main/alex20s/csp/epoch_046000.pkl)

## Model Architecture

Both checkpoints share the same Transformer architecture:

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

**[Alex-20s](https://huggingface.co/datasets/zdcao/alex-20s)**: ~1.7M general inorganic materials curated from the [Alexandria database](https://alexandria.icams.rub.de/), filtered by:
- Energy above hull: $E_{hull} < 0.1$ eV/atom
- Structure complexity: no more than 20 Wyckoff sites in the conventional cell

## Speeds, Sizes, Times
- Both models contain ~13.8M parameters
- Generating 29,000 crystal samples on a single A100 GPU takes ~1,058 seconds (~37 ms per sample)