# TinyTransformer Lab

A compact educational character-level Transformer written in PyTorch. It trains on a small text corpus, generates text autoregressively, and saves a loss curve. The code favors readability over scale so every part of the training loop is easy to inspect.

## Run

```bash
pip install -r requirements.txt
python -m tinytransformer --steps 300 --prompt "AI "
```

Run tests with `python -m unittest discover -s tests`.

## Learning goals

- Character tokenization and causal language modeling
- Token and positional embeddings
- Multi-head self-attention with a causal mask
- Teacher forcing and autoregressive sampling
- Reproducible experiments and loss plots
