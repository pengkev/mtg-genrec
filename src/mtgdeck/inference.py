"""Shared inference adapter for Oracle-ID VAE deck completion checkpoints.

Parsing, model construction and ranking preserve the original demo semantics.
No UI, training corpus or Card2Vec training package is needed at inference time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import torch


from .data import ORACLE_TOKEN_PREFIX, PAD_TOKEN, UNK_TOKEN, deck_to_tokens
from .legality import CommanderCandidateIndex, OracleCatalog
from .vae import Card2VecAttentionVAE, collate_token_rows, mask_present_logits


IGNORED_SECTIONS = {"sideboard", "sideboard:", "maybeboard", "maybeboard:"}
COMMANDER_SECTIONS = {"commander", "commander:", "commanders", "commanders:"}
MAINBOARD_SECTIONS = {"deck", "deck:", "mainboard", "mainboard:"}
SET_SUFFIX = re.compile(r"\s+\([A-Za-z0-9]{2,8}\)\s+[A-Za-z0-9-]+\s*$")
QUANTITY_PREFIX = re.compile(r"^\s*(\d+)\s*(?:[xX]\s*)?(.+?)\s*$")


def available_checkpoints(checkpoint_dir: Path) -> list[Path]:
    preferred = [
        checkpoint_dir / "attention_oracleid_v2_variational_finetuned_896.pt",
        checkpoint_dir / "attention_oracleid_v2_variational_frozen_896.pt",
        checkpoint_dir / "attention_oracleid_v2_deterministic_frozen_896.pt",
    ]
    discovered = sorted(checkpoint_dir.glob("attention_oracleid_v2_*_896.pt"))
    return [path for path in dict.fromkeys([*preferred, *discovered]) if path.exists()]


def _checkpoint_label(path: Path) -> str:
    name = path.stem.replace("attention_oracleid_v2_", "").replace("_896", "")
    labels = {
        "variational_finetuned": "Variational · fine-tuned Card2Vec (best NDCG)",
        "variational_frozen": "Variational · frozen Card2Vec",
        "deterministic_frozen": "Deterministic · frozen Card2Vec (best Recall@20)",
    }
    return labels.get(name, name.replace("_", " · ").title())


def _parse_line(raw_line: str) -> tuple[int, str] | None:
    line = raw_line.strip()
    if not line or line.startswith(("#", "//")):
        return None
    line = re.sub(r"^SB:\s*", "", line, flags=re.IGNORECASE)
    match = QUANTITY_PREFIX.match(line)
    if match:
        return max(1, int(match.group(1))), match.group(2).strip()
    return 1, line


def parse_deck_text(text: str, initial_zone: str = "mainboard") -> dict[str, Counter[str]]:
    """Parse common Arena/Moxfield-style text into commander/mainboard counters."""

    result = {"commanders": Counter(), "mainboard": Counter()}
    zone = initial_zone
    for raw_line in text.splitlines():
        heading = raw_line.strip().casefold()
        if heading in COMMANDER_SECTIONS:
            zone = "commanders"
            continue
        if heading in MAINBOARD_SECTIONS:
            zone = "mainboard"
            continue
        if heading in IGNORED_SECTIONS:
            zone = "ignore"
            continue
        parsed = _parse_line(raw_line)
        if parsed is None or zone == "ignore":
            continue
        quantity, name = parsed
        result[zone][name] += quantity
    return result


def _resolve_name(catalog: OracleCatalog, raw_name: str) -> Mapping[str, Any] | None:
    card = catalog.resolve(raw_name)
    if card is not None:
        return card
    stripped = SET_SUFFIX.sub("", raw_name).strip()
    return catalog.resolve(stripped) if stripped != raw_name else None


def _build_model(checkpoint: Mapping[str, Any], device: torch.device) -> Card2VecAttentionVAE:
    state = checkpoint["state_dict"]
    config = checkpoint.get("config", {})
    embeddings = state["card_embedding.weight"].detach().cpu()
    variational = bool(config.get("variational", "deterministic" not in str(config.get("experiment", ""))))
    model = Card2VecAttentionVAE(
        embeddings,
        model_dim=int(config.get("model_dim", 256)),
        num_heads=int(config.get("heads", 8)),
        num_layers=int(config.get("blocks", 2)),
        latent_dim=int(config.get("latent_dim", 64)),
        num_pool_queries=int(config.get("pool_queries", 4)),
        num_decoder_queries=int(config.get("decoder_queries", 4)),
        freeze_card2vec=True,
        initial_logit_scale=float(config.get("initial_logit_scale", 10.0)),
        variational=variational,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_bundle(
    checkpoint_path: str, oracle_path: str, device_name: str,
    eligibility_path: str | None = None, catalog: OracleCatalog | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    if not any(token.startswith(ORACLE_TOKEN_PREFIX) for token in vocab):
        raise ValueError("This demo requires an oracleid_v2 checkpoint.")
    catalog = catalog if catalog is not None else OracleCatalog.from_path(oracle_path, eligibility_path)
    candidate_index = CommanderCandidateIndex(catalog, vocab)
    return {
        "checkpoint": checkpoint,
        "model": _build_model(checkpoint, device),
        "vocab": vocab,
        "inverse_vocab": {index: token for token, index in vocab.items()},
        "catalog": catalog,
        "candidate_index": candidate_index,
        "token_cards": candidate_index.token_cards,
        "device": device,
    }


def resolve_partial_deck(
    catalog: OracleCatalog,
    vocab: Mapping[str, int],
    commander_text: str,
    deck_text: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    commander_entries = parse_deck_text(commander_text, "commanders")["commanders"]
    parsed_deck = parse_deck_text(deck_text, "mainboard")
    commander_entries.update(parsed_deck["commanders"])
    mainboard_entries = parsed_deck["mainboard"]

    unresolved: list[str] = []
    out_of_vocabulary: list[str] = []

    def resolve_zone(entries: Iterable[tuple[str, int]]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for raw_name, quantity in entries:
            card = _resolve_name(catalog, raw_name)
            if card is None:
                unresolved.append(raw_name)
                continue
            token = ORACLE_TOKEN_PREFIX + str(card["oracle_id"]).casefold()
            if token not in vocab:
                out_of_vocabulary.append(str(card["name"]))
            resolved.append(
                {
                    "name": token,
                    "quantity": int(quantity),
                    "oracle_id": str(card["oracle_id"]),
                    "display_name": str(card["name"]),
                }
            )
        return resolved

    def merge_identities(items: list[dict[str, Any]], commander: bool = False) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in items:
            oracle_id = item["oracle_id"]
            if oracle_id not in merged:
                merged[oracle_id] = dict(item)
            elif not commander:
                merged[oracle_id]["quantity"] += item["quantity"]
        if commander:
            for item in merged.values():
                item["quantity"] = 1
        return list(merged.values())

    commanders = merge_identities(resolve_zone(commander_entries.items()), commander=True)
    mainboard = merge_identities(resolve_zone(mainboard_entries.items()))
    commander_ids = {item["oracle_id"] for item in commanders}
    mainboard = [item for item in mainboard if item["oracle_id"] not in commander_ids]
    partial = {
        "schema_version": 1,
        "deck_id": "demo:partial",
        "source": "demo",
        "source_id": "partial",
        "url": None,
        "name": "Interactive partial deck",
        "format": "commander",
        "date": None,
        "commanders": commanders,
        "mainboard": mainboard,
        "sideboard": [],
        "metadata": {},
    }
    return partial, unresolved, out_of_vocabulary


@torch.inference_mode()
def recommend(
    bundle: Mapping[str, Any],
    partial: Mapping[str, Any],
    count: int,
    sample_latent: bool,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    model = bundle["model"]
    vocab = bundle["vocab"]
    device = bundle["device"]
    token_rows = [deck_to_tokens(partial, vocab)]
    if not token_rows[0][0]:
        raise ValueError("None of the supplied cards occur in the checkpoint vocabulary.")
    ids, roles, quantities, padding = (
        tensor.to(device)
        for tensor in collate_token_rows(token_rows, vocab.get(PAD_TOKEN, 0))
    )
    allowed = torch.as_tensor(
        bundle["candidate_index"].allowed_mask(partial),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    accumulated = None
    actual_draws = max(1, draws if sample_latent and model.variational else 1)
    for _ in range(actual_draws):
        outputs = model(ids, roles, quantities, padding, sample=sample_latent)
        scores = mask_present_logits(outputs["logits"], ids, padding).masked_fill(
            ~allowed, -torch.inf
        )
        accumulated = scores if accumulated is None else accumulated + scores
    scores = accumulated / actual_draws
    for special in (PAD_TOKEN, UNK_TOKEN):
        if special in vocab:
            scores[:, vocab[special]] = -torch.inf

    finite_count = int(torch.isfinite(scores[0]).sum().item())
    requested = min(max(1, int(count)), finite_count)
    values, indices = torch.topk(scores[0], requested)
    rows = []
    for rank, (value, index) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), 1):
        token = bundle["inverse_vocab"][index]
        card = bundle["token_cards"][token]
        rows.append(
            {
                "Rank": rank,
                "Card": card["name"],
                "Score": round(float(value), 4),
                "Color identity": "".join(card.get("color_identity") or []) or "Colorless",
                "Type": card.get("type_line", ""),
            }
        )
    return rows


RECOMMENDATION_COLUMNS = ["Rank", "Card", "Score", "Color identity", "Type"]
VISIBLE_COLUMNS = ["Zone", "Quantity", "Card"]
DEFAULT_COMMANDER = "Muldrotha, the Gravetide"
DEFAULT_DECK = """1 Sol Ring
1 Arcane Signet
1 Command Tower
1 Cultivate
1 Sakura-Tribe Elder
1 Eternal Witness
1 Mulldrifter
1 Counterspell
1 Putrefy
1 Lightning Greaves"""


@dataclass
class PreparedRequest:
    partial: dict[str, Any]
    visible: list[list[Any]]
    notices: list[str]
    count: int
    sample_latent: bool
    draws: int
    seed: int


def prepare_request(
    bundle: Mapping[str, Any], commander_text: str, deck_text: str,
    count: int = 25, sample_latent: bool = False, draws: int = 1, seed: int = 42,
) -> PreparedRequest:
    """Validate public UI/API controls and resolve names before model inference."""
    for name, value, minimum, maximum in (
        ("Recommendations", count, 5, 100), ("Latent draws", draws, 1, 16),
        ("Sampling seed", seed, 0, 2_147_483_647),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum or int(value) != value:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    partial, unresolved, out_of_vocabulary = resolve_partial_deck(
        bundle["catalog"], bundle["vocab"], commander_text, deck_text,
    )
    notices = []
    if unresolved:
        notices.append("Could not resolve: " + ", ".join(sorted(set(unresolved))))
    if not partial["commanders"]:
        raise ValueError("Enter at least one resolvable Commander.")
    if out_of_vocabulary:
        notices.append("Resolved but absent from the training vocabulary: " + ", ".join(sorted(set(out_of_vocabulary))))
    visible = [
        ["Commander" if zone == "commanders" else "Mainboard", item["quantity"], item["display_name"]]
        for zone in ("commanders", "mainboard") for item in partial[zone]
    ]
    return PreparedRequest(partial, visible, notices, int(count), bool(sample_latent), int(draws), int(seed))
