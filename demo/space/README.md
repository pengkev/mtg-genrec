---
title: Mtg Genrec
emoji: 🦀
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: An approach to card recommendation for decks in MTG
---

# MTG GenRec

Commander deck completion using Card2Vec and an attention VAE.
Paste a commander and partial deck to rank legal missing cards. Scores are
relative model logits, not probabilities or power ratings.

Source of truth: [GitHub main](https://github.com/pengkev/mtg-genrec).
This Space is a generated deployment artifact. Edit code and configuration in
GitHub; do not edit the Space independently. `source.json` records the source
commit and `artifacts.json` pins the inference assets by SHA-256.

The hosted app detects ZeroGPU hardware, places models on CUDA at startup and
decorates only scoring with `spaces.GPU(duration=5)`. No placeholder GPU work
or per-request model loading is used.

Local runs default to CPU: two-thread inference measured 18–31 ms normally and
about 275 ms for 16 latent draws. CPU is fast enough, but this Space's ZeroGPU
hardware requires a GPU function and the account could not switch it to free
CPU hardware. The existing ZeroGPU hardware therefore serves actual inference.
