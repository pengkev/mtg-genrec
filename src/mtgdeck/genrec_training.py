"""Format-specific GenRec preparation and two-stage training used by the notebook."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
from itertools import islice
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import (ORACLE_TOKEN_PREFIX, PAD_TOKEN, UNK_TOKEN, build_vocabulary,
                   canonical_validation_errors, deck_to_tokens,
                   iter_jsonl, mask_deck)
from .genrec_embeddings import align_static_embeddings, genrec_optimizer
from .legality import CommanderCandidateIndex, allowed_copy_count, validate_commander_deck
from .recommend import deck_card_names, ndcg_at_k, recall_at_k
from .vae import (Card2VecAttentionVAE, collate_token_rows, kl_beta,
                  mask_present_logits, score_cards_batch, vae_loss)

FORMATS = ('commander', 'modern', 'legacy')


def canonical_format(value):
    value = str(value).strip().casefold()
    return 'commander' if value in {'commander', 'edh', 'cedh'} else value


class FormatCandidateIndex:
    """Snapshot legality; only Commander applies commander/color constraints."""
    def __init__(self, catalog, vocab, format_name):
        self.format = canonical_format(format_name)
        if self.format not in FORMATS:
            raise ValueError(f'Unsupported format: {format_name}')
        self.commander = CommanderCandidateIndex(catalog, vocab) if self.format == 'commander' else None
        self.mask = np.zeros(len(vocab), dtype=bool)
        for token, index in vocab.items():
            if token in {PAD_TOKEN, UNK_TOKEN}:
                continue
            card = catalog.resolve('', token.removeprefix(ORACLE_TOKEN_PREFIX))
            if card is None:
                raise ValueError(f'Unknown Oracle identity: {token}')
            self.mask[index] = card.get('legalities', {}).get(self.format) == 'legal'

    def allowed_mask(self, deck):
        return self.commander.allowed_mask(deck) if self.commander else self.mask.copy()


def canonical_model_deck(record, catalog, format_name):
    """Validate against the local snapshot, preserving quantities and provenance."""
    fmt = canonical_format(format_name)
    if fmt not in FORMATS or canonical_format(record.get('format')) != fmt:
        raise ValueError('wrong_format')
    if canonical_validation_errors(record):
        raise ValueError('invalid_schema')
    record = copy.deepcopy(record)
    record['format'] = fmt
    if fmt == 'commander':
        result = validate_commander_deck(record, catalog)
        if not result.legal:
            raise ValueError(','.join(sorted(result.reason_codes)))
        record = result.cleaned_deck
    elif record.get('commanders'):
        raise ValueError('unexpected_command_zone')
    cards_by_id = {}
    for zone in ('commanders', 'mainboard', 'sideboard'):
        merged = {}
        for item in record.get(zone, []):
            card = catalog.resolve(item['name'], item.get('oracle_id'))
            if card is None:
                raise ValueError('unknown_card')
            if item.get('oracle_id') and not catalog.name_matches(card, item['name']):
                raise ValueError('oracle_name_mismatch')
            if card.get('legalities', {}).get(fmt) != 'legal':
                raise ValueError('card_not_format_legal')
            oid = str(card['oracle_id']).casefold()
            cards_by_id[oid] = card
            row = merged.setdefault(oid, {'name': ORACLE_TOKEN_PREFIX + oid,
                                         'oracle_id': oid, 'quantity': 0})
            row['quantity'] += item['quantity']
        record[zone] = list(merged.values())
    if fmt != 'commander':
        if sum(x['quantity'] for x in record['mainboard']) < 60:
            raise ValueError('mainboard_below_60')
        if sum(x['quantity'] for x in record['sideboard']) > 15:
            raise ValueError('sideboard_above_15')
        copies = Counter()
        for zone in ('mainboard', 'sideboard'):
            copies.update({x['oracle_id']: x['quantity'] for x in record[zone]})
        for oid, quantity in copies.items():
            limit = allowed_copy_count(cards_by_id[oid])
            if limit is not None and quantity > max(4, limit):
                raise ValueError('copy_limit')
    if len(record['mainboard']) < 2:
        raise ValueError('insufficient_mainboard_identities')
    return record


def load_format_decks(paths, catalog, format_name, max_records=None):
    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        raise FileNotFoundError(f'No {format_name} input exists: {paths}')
    audit = {'files': [str(p) for p in existing], 'read': 0, 'accepted': 0,
             'rejected': Counter(), 'examples': []}
    def records():
        for path in existing:
            yield from iter_jsonl(path)
    decks = []
    for record in islice(records(), max_records):
        audit['read'] += 1
        try:
            deck = canonical_model_deck(record, catalog, format_name)
        except ValueError as error:
            audit['rejected'][str(error)] += 1
            if len(audit['examples']) < 20:
                audit['examples'].append({'deck_id': record.get('deck_id'), 'reason': str(error)})
            continue
        decks.append(deck)
    audit['accepted'] = len(decks)
    return decks, audit


def split_joint_corpora(base, premium, seed=42, threshold=.90, components=128, band_size=8):
    """Joint MinHash candidate grouping + exact Jaccard verification across tiers.

    Identical model-visible card sets are deduplicated, retaining premium origin.
    Quantities remain model inputs, but changing copies/sideboards cannot move
    the same mainboard identities into a different split. Premium-bearing groups
    are assigned first to balance premium holdouts, then base-only groups.
    """
    if not 0 < threshold <= 1 or components < 1 or band_size < 1 or components % band_size:
        raise ValueError('Invalid near-duplicate grouping parameters')
    unique, identities = {}, defaultdict(set)
    for tier, rows in (('base', base), ('premium', premium)):
        for deck in rows:
            key = tuple(sorted(deck_card_names(deck)))
            unique[key] = (tier, deck)
            identities[key].add((deck.get('source'), deck.get('source_id') or deck['deck_id']))
    entries = list(unique.values())
    sets = [set(key) for key in unique]
    parent = list(range(len(entries)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    buckets, hashes, source_ids = defaultdict(list), {}, {}
    for i, cards in enumerate(sets):
        for identity in sorted(identities[tuple(sorted(cards))], key=str):
            if identity in source_ids:
                parent[find(i)] = find(source_ids[identity])
            source_ids[identity] = i
        for card in cards:
            if card not in hashes:
                hashes[card] = np.array([int.from_bytes(hashlib.blake2b(
                    j.to_bytes(2, 'little') + card.encode(), digest_size=8).digest(), 'little')
                    for j in range(components)], dtype=np.uint64)
        signature = np.stack([hashes[c] for c in cards]).min(axis=0)
        keys = [(start, signature[start:start + band_size].tobytes())
                for start in range(0, components, band_size)]
        candidates = set(j for key in keys for j in buckets[key])
        for j in candidates:
            if min(len(cards), len(sets[j])) / max(len(cards), len(sets[j])) < threshold:
                continue
            if len(cards & sets[j]) / len(cards | sets[j]) >= threshold:
                parent[find(i)] = find(j)
        for key in keys:
            buckets[key].append(i)
    grouped = defaultdict(list)
    for i in range(len(entries)):
        grouped[find(i)].append(i)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: not any(entries[i][0] == 'premium' for i in group))
    splits = {tier: {name: [] for name in ('train', 'validation', 'test')} for tier in ('base', 'premium')}
    membership = []
    for group in groups:
        balance_tier = 'premium' if any(entries[i][0] == 'premium' for i in group) else 'base'
        destination = min(zip(('train', 'validation', 'test'), (.8, .1, .1)),
                          key=lambda pair: len(splits[balance_tier][pair[0]]) / pair[1])[0]
        for i in group:
            tier, deck = entries[i]
            splits[tier][destination].append(deck)
            membership.append({'tier': tier, 'split': destination, 'deck_id': deck['deck_id'],
                               'group': find(i), 'card_set_sha256': hashlib.sha256(
                                   '\n'.join(sorted(sets[i])).encode()).hexdigest()})
    if any(not rows for partitions in splits.values() for rows in partitions.values()):
        raise ValueError('Need nonempty base and premium train/validation/test splits after joint grouping')
    return splits, {'groups': len(groups), 'deduplicated': len(base) + len(premium) - len(entries),
                    'threshold': threshold, 'minhash_components': components, 'band_size': band_size,
                    'sizes': {tier: {name: len(rows) for name, rows in parts.items()} for tier, parts in splits.items()},
                    'membership': membership}


def prepare_format(format_name, paths, catalog, bundle, config):
    base, base_audit = load_format_decks(paths['base'], catalog, format_name, config.get('max_records'))
    premium, premium_audit = load_format_decks(paths['premium'], catalog, format_name, config.get('max_records'))
    splits, split_audit = split_joint_corpora(base, premium, config['seed'], config['near_duplicate_threshold'])
    # Premium train can introduce cards absent from base train; never use holdouts.
    vocab = build_vocabulary(splits['base']['train'] + splits['premium']['train'], min_count=1)
    matrix, coverage = align_static_embeddings(bundle, vocab, catalog,
        missing_policy=config['missing_embedding_policy'], seed=config['seed'])
    return {'format': canonical_format(format_name), 'splits': splits, 'vocab': vocab,
            'weights': torch.from_numpy(matrix), 'candidates': FormatCandidateIndex(catalog, vocab, format_name),
            'audit': {'base': base_audit, 'premium': premium_audit, 'splits': split_audit, 'embeddings': coverage}}


def make_model(prepared, config):
    torch.manual_seed(config['seed'])
    return Card2VecAttentionVAE(prepared['weights'].clone(), model_dim=config['model_dim'],
        num_heads=config['heads'], num_layers=config['blocks'], latent_dim=config['latent_dim'],
        num_pool_queries=config['pool_queries'], num_decoder_queries=config['decoder_queries'],
        freeze_card2vec=False, variational=True).to(config['device'])


def masked_examples(rows, ratio, seed, repeat=0):
    for i, deck in enumerate(rows):
        visible, hidden = mask_deck(deck, ratio, seed + i + repeat * 1_000_003)
        yield visible, [x['name'] for x in hidden]


@torch.no_grad()
def evaluate_model(model, rows, prepared, config):
    vocab = prepared['vocab']
    inverse = {index: name for name, index in vocab.items()}
    totals = Counter()
    tasks = hidden_total = covered = 0
    for repeat in range(config['eval_repeats']):
        examples = list(masked_examples(rows, config['eval_mask_ratio'], config['seed'], repeat))
        for start in range(0, len(examples), config['eval_batch_size']):
            batch = examples[start:start + config['eval_batch_size']]
            visible = [x[0] for x in batch]
            allowed = np.stack([prepared['candidates'].allowed_mask(deck) for deck in visible])
            scores = score_cards_batch(model, visible, vocab, config['device'], allowed)
            values, indices = scores.topk(min(50, len(vocab)), dim=1)
            for values_row, ids, (_, hidden) in zip(values.cpu(), indices.cpu().tolist(), batch):
                ranked = [inverse[i] for value, i in zip(values_row, ids) if torch.isfinite(value)]
                for k in (10, 20, 50):
                    totals[f'Recall@{k}'] += recall_at_k(ranked, hidden, k)
                    totals[f'NDCG@{k}'] += ndcg_at_k(ranked, hidden, k)
                hidden_total += len(hidden)
                covered += sum(name in vocab for name in hidden)
                tasks += 1
    if not tasks:
        raise ValueError('Cannot evaluate empty split')
    return {'tasks': tasks, 'target_vocab_coverage': covered / max(1, hidden_total),
            **{key: value / tasks for key, value in totals.items()}}


def train_stage(model, prepared, config, stage, checkpoint, parent_checkpoint=None):
    """Base -> best base -> premium, with separate rates and stage validation."""
    if stage not in ('base', 'premium'):
        raise ValueError(stage)
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_config = config[stage]
    if stage_config['epochs'] < 1:
        raise ValueError('Each stage must have at least one training epoch')
    model.card_embedding.weight.requires_grad_(True)
    optimizer = genrec_optimizer(model, stage_config['learning_rate'], stage_config['embedding_learning_rate'])
    torch.manual_seed(config['seed'])
    rows = prepared['splits'][stage]['train']
    validation_rows = prepared['splits'][stage]['validation']
    vocab = prepared['vocab']
    best_score = -float('inf')
    history, step = [], 0
    # Stage two must improve on the incoming base model on premium validation;
    # the epoch-zero candidate prevents silently replacing it with a worse fit.
    def save(epoch, metrics):
        serial_config = {**config, 'format': prepared['format'], 'stage': stage,
                         'variational': True, 'freeze_card2vec': False,
                         'embedding_provenance': prepared['audit']['embeddings']}
        torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'config': serial_config,
                    'epoch': epoch, 'val_metrics': metrics,
                    'parent_checkpoint': str(parent_checkpoint) if parent_checkpoint else None}, checkpoint)
    if stage == 'premium':
        if parent_checkpoint is None:
            raise ValueError('Premium stage requires a base checkpoint')
        parent = torch.load(parent_checkpoint, map_location=config['device'], weights_only=True)
        if parent['vocab'] != vocab or parent['config']['format'] != prepared['format'] or parent['config']['stage'] != 'base':
            raise ValueError('Premium parent must be the same-format base model and vocabulary')
        model.load_state_dict(parent['state_dict'])
        initial = evaluate_model(model, validation_rows, prepared, config)
        best_score = initial['Recall@20']
        save(0, initial)
        history.append({'epoch': 0, 'validation': initial})
    for epoch in range(stage_config['epochs']):
        model.train()
        shuffled = list(rows)
        random.Random(config['seed'] + epoch).shuffle(shuffled)
        losses = []
        for start in range(0, len(shuffled), config['batch_size']):
            visible, hidden = [], []
            for i, deck in enumerate(shuffled[start:start + config['batch_size']]):
                seed = config['seed'] + epoch * len(rows) + start + i
                ratio = random.Random(seed).choice(config['train_mask_ratios'])
                partial, targets = mask_deck(deck, ratio, seed)
                visible.append(partial)
                hidden.append(targets)
            ids, roles, qty, padding = [x.to(config['device']) for x in collate_token_rows(
                [deck_to_tokens(deck, vocab) for deck in visible])]
            targets = torch.zeros(len(visible), len(vocab), device=config['device'])
            for i, cards in enumerate(hidden):
                for card in cards:
                    targets[i, vocab[card['name']]] = 1
            allowed = torch.as_tensor(np.stack([prepared['candidates'].allowed_mask(d) for d in visible]), device=config['device'])
            output = model(ids, roles, qty, padding)
            output['logits'] = mask_present_logits(output['logits'], ids, padding).masked_fill(~allowed, -torch.inf)
            if not torch.isfinite(output['logits'][targets.bool()]).all():
                raise ValueError('Hidden positive is illegal, visible, or nonfinite')
            beta = kl_beta(step, stage_config['kl_warmup_steps'], config['kl_beta'])
            total, reconstruction, kl = vae_loss(output, targets, beta)
            if not all(torch.isfinite(x) for x in (total, reconstruction, kl)):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append([total.item(), reconstruction.item(), kl.item(), beta])
            step += 1
        metrics = evaluate_model(model, validation_rows, prepared, config)
        row = {'epoch': epoch + 1, 'loss': np.mean(losses, axis=0).tolist(), 'validation': metrics}
        history.append(row)
        print(prepared['format'], stage, row, flush=True)
        if metrics['Recall@20'] > best_score:
            best_score = metrics['Recall@20']
            save(epoch + 1, metrics)
    best = torch.load(checkpoint, map_location=config['device'], weights_only=True)
    model.load_state_dict(best['state_dict'])
    return {'checkpoint': str(checkpoint), 'best_epoch': best['epoch'], 'history': history,
            'validation': best['val_metrics']}
