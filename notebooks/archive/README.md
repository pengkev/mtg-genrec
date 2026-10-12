# Historical notebooks

The active notebooks contain only the current workflows:

- [`../card2vec_static_embeddings.ipynb`](../card2vec_static_embeddings.ipynb): cached-data readiness, seven-stage execution, and saved-result review.
- [`../genrec.ipynb`](../genrec.ipynb): format-specific base training, premium refinement, evaluation, and recommendations.

Historical cells and saved outputs remain here:

- [`card2vec_static_embeddings.ipynb`](card2vec_static_embeddings.ipynb): dimension, geometry, relation, fusion, and earlier v4 experiments.
- [`genrec_commander_comparisons.ipynb`](genrec_commander_comparisons.ipynb): saved EDHREC and historical 896d-model comparisons.

Open these for reference or run a selected section. The Card2Vec archive combines
separate experiments; use their targeted CLI stages instead of Run All.
`scripts/render_static_review.py`, `scripts/render_static_fusion_review.py`, and
`scripts/run_static_geometry.py --stage render` read the archive.
`scripts/render_static_v4_review.py` reviews the active pipeline without running it.

Archive relocation preserves saved outputs and adjusts only repository discovery
in the setup cells. Frozen training implementations and existing artifacts stay
unchanged.
