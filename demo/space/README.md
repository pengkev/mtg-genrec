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

CPU is the default: local two-thread inference measured 18–31 ms for a normal
request and about 275 ms for 16 latent draws. This avoids a GPU allocation for
an already interactive workload. The app also supports `MTG_DEVICE=zerogpu`
on ZeroGPU hardware, with models placed on CUDA at startup and only scoring
decorated with `spaces.GPU(duration=5)`.
