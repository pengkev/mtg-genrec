"""Oracle corpus, auditable synthetic relations, and opt-in ModernBERT stages.

Model libraries are imported only inside explicit training/extraction functions.
All model loading is local-only; download_model is the only model network stage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re

import numpy as np

from .data import normalize_card_name
from .metadata import iter_oracle_cards
from .static_export import unit_rows
from . import static_v4 as v4

MODEL = 'answerdotai/ModernBERT-base'
VIEWS = ('rules_only', 'structured_no_name', 'normalized_semantic')
FIELD_TOKENS = ('[CARD]', '[MANA]', '[TYPE]', '[TEXT]', '[PT]', '[LOYALTY]',
                '[DEFENSE]', '[FACE]', '[SELF]', '[COST]', '[ACTION]', '[TARGET]',
                '[RESTRICTION]', '[RESOURCE]', '[SPEED]', '[SECONDARY]', '[REPLACEMENT]')
ANCHORS = ('Remand', 'Reprieve', 'Lightning Bolt', 'Wrath of God', 'Llanowar Elves',
           'Soulherder', 'Young Pyromancer', 'Entomb')
ACTION_PATTERNS = {
    'counter': r'\bcounter target (?:\w+ )*spell', 'destroy': r'\bdestroy\b',
    'exile': r'\bexile\b', 'return': r'\breturn\b', 'draw': r'\bdraw\b',
    'discard': r'\bdiscard\b', 'search': r'\bsearch\b', 'add_mana': r'\badd (?:\{|\w+ mana)',
    'copy': r'\bcopy\b', 'sacrifice': r'\bsacrifice\b', 'create_token': r'\bcreate\b[^.]*\btoken',
    'mill': r'\bmill\b', 'damage': r'\bdeals?\b[^.]*\bdamage\b',
    'gain_life': r'\bgain\b[^.]*\blife\b',
    'blink': r'\bexile\b[^.]*\breturn\b',
    'reanimate': r'\breturn\b[^.]*\bgraveyard\b[^.]*\bbattlefield\b',
}
RESTRICTIONS = {
    'nonland': r'\bnonland\b', 'noncreature': r'\bnoncreature\b',
    'mana_value_bound': r'\bmana value\b[^.]*?(?:less|greater|equal)',
    'color_restriction': r'\b(?:white|blue|black|red|green|colorless|multicolored) (?:spell|creature|permanent)',
    'tapped': r'\btapped\b', 'untapped': r'\buntapped\b',
    'attacking': r'\battacking\b', 'blocking': r'\bblocking\b',
    'controller': r'\b(?:you|they) control\b', 'owner': r"\bowner'?s?\b",
    'once_per_turn': r'\bonly once each turn\b', 'timing_condition': r'\b(?:only during|only as a sorcery)\b',
    'conditional': r'\b(?:if|unless)\b', 'symmetric': r'\b(?:all|each) (?:creature|player|permanent)',
}


def faces(card):
    return card.get('card_faces') or [card]


def rules_text(card):
    # Remove self-name identity, including back-face names; never include flavor text.
    text = '\n'.join(str(face.get('oracle_text', '')) for face in faces(card))
    for name in sorted({card.get('name', ''), *(f.get('name', '') for f in faces(card))}, key=len, reverse=True):
        if name:
            text = re.sub(r'(?<!\w)' + re.escape(name) + r'(?!\w)', '[SELF]', text, flags=re.I)
    return text.strip()


def effect_signature(card):
    """Lexical evidence, not a Comprehensive Rules interpreter.

    Every tag has the exact matched source span. Targets are lexical mentions,
    not inferred scope; conservative tier-2 proposals remain filterable.
    """
    text = rules_text(card).lower()
    evidence = {}
    def extract(patterns, field):
        found = []
        for tag, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                found.append(tag)
                evidence[f'{field}:{tag}'] = match.group()
        return sorted(found)
    actions = extract(ACTION_PATTERNS, 'action')
    targets = extract({tag: rf'\b{tag}\b' for tag in
                       ('spell', 'creature', 'permanent', 'artifact', 'enchantment', 'planeswalker', 'player')}, 'target')
    if re.search(r'\bcard\b[^.]*\bgraveyard\b', text):
        targets.append('graveyard_card')
        evidence['target:graveyard_card'] = re.search(r'\bcard\b[^.]*\bgraveyard\b', text).group()
    restrictions = extract(RESTRICTIONS, 'restriction')
    speed = []
    for kind in ('instant', 'sorcery'):
        if kind in card.get('type_line', '').lower():
            speed.append(kind)
            evidence[f'speed:{kind}'] = card['type_line']
    speed += extract({'activated': r':', 'triggered': r'\b(?:when|whenever|at the beginning)\b',
                      'static': r'\b(?:have|has|gets?|costs?)\b'}, 'speed')
    resource = extract({'cards': r'\b(?:draw|discard)\b', 'mana': r'\b(?:mana|add \{)\b',
                        'life': r'\blife\b', 'treasure': r'\btreasure\b', 'tokens': r'\btokens?\b'}, 'resource')
    secondary = extract({'draw': r'\bdraw\b', 'life_loss': r'\blose\b[^.]*\blife\b',
                         'self_exile': r'\bexile \[self\]', 'tokens': r'\bcreate\b[^.]*\btoken',
                         'recursion': r'\breturn\b[^.]*\bgraveyard\b'}, 'secondary')
    replacement = extract({'instead': r'\binstead\b', 'return_to_hand': r'\b(?:return|put)\b[^.]*\bhand\b'}, 'replacement')
    return dict(actions=actions, targets=sorted(targets), restrictions=restrictions,
                speed=sorted(speed), resources=resource, secondary=secondary,
                replacement=replacement, mana_value=card.get('cmc'), evidence=evidence)


def serialize(card, view='structured_no_name'):
    if view == 'rules_only':
        return rules_text(card)
    if view == 'normalized_semantic':
        sig = effect_signature(card)
        lines = ['[TYPE] ' + card.get('type_line', '').lower()]
        if sig['mana_value'] is not None:
            lines.append('[COST] ' + str(sig['mana_value']))
        for field, marker in [('actions', 'ACTION'), ('targets', 'TARGET'), ('restrictions', 'RESTRICTION'),
                              ('resources', 'RESOURCE'), ('speed', 'SPEED'), ('secondary', 'SECONDARY'),
                              ('replacement', 'REPLACEMENT')]:
            if sig[field]:
                lines.append(f'[{marker}] ' + ', '.join(sig[field]))
        return '\n'.join(lines)
    if view != 'structured_no_name':
        raise ValueError(f'Unknown serialization: {view}')
    lines = ['[CARD]']
    for i, face in enumerate(faces(card)):
        if i:
            lines.append('[FACE]')
        for field, marker in [('mana_cost', 'MANA'), ('type_line', 'TYPE'), ('oracle_text', 'TEXT')]:
            if face.get(field):
                value = rules_text({**face, 'name': face.get('name', card.get('name', ''))}) if field == 'oracle_text' else face[field]
                lines.append(f'[{marker}] {value}')
        if 'power' in face and 'toughness' in face:
            lines.append(f"[PT] {face['power']}/{face['toughness']}")
        for field in ('loyalty', 'defense'):
            if field in face:
                lines.append(f'[{field.upper()}] {face[field]}')
    return '\n'.join(lines)


def discover_tokens(cards):
    symbols = set()
    markers = set()
    for card in cards:
        for view in VIEWS:
            text = serialize(card, view)
            symbols.update(re.findall(r'\{[^{}\s]+\}', text))
            markers.update(set(re.findall(r'\[[A-Z_]+\]', text)) & set(FIELD_TOKENS))
    return {'symbols': sorted(symbols), 'fields': sorted(markers)}


def normalized_rules(card):
    return re.sub(r'\s+', ' ', rules_text(card).casefold()).strip()


def description(card):
    # Exact Oracle wording supplies effect detail; no unsupported paraphrasing.
    cost = f"{card['cmc']:g}-mana " if isinstance(card.get('cmc'), (float, int)) else ''
    return f"{cost}{card.get('type_line', 'card').lower()} with these rules: {rules_text(card)}"


def hard_negative_reasons(a, b):
    reasons = []
    if a['targets'] != b['targets']:
        reasons.append('same_role_wrong_target')
    if ('draw' in a['secondary']) != ('draw' in b['secondary']):
        reasons.append('same_effect_missing_cantrip')
    if ({'destroy', 'exile'} <= set(a['actions'] + b['actions']) and a['actions'] != b['actions']):
        reasons.append('destroy_vs_exile')
    if ('return_to_hand' in a['replacement']) != ('return_to_hand' in b['replacement']):
        reasons.append('temporary_vs_permanent_answer')
    if a['mana_value'] is not None and b['mana_value'] is not None and abs(a['mana_value'] - b['mana_value']) >= 2:
        reasons.append('mana_cost_mismatch')
    if a['speed'] != b['speed']:
        reasons.append('timing_mismatch')
    if a['restrictions'] != b['restrictions']:
        reasons.append('restriction_mismatch')
    if a['replacement'] != b['replacement']:
        reasons.append('replacement_mismatch')
    return reasons


def synthetic_dataset(cards, curated=None, reserved=None, replacements=(), seed=42):
    """Family-disjoint positives and split-local negatives; no probe supervision."""
    lookup = {c['canonical']: c for c in cards}
    reserved = set(reserved or ())
    families = v4.Families()
    groups = defaultdict(list)
    signatures = {n: effect_signature(c) for n, c in lookup.items()}
    for name, card in lookup.items():
        families.find(name)
        text = normalized_rules(card)
        if text:
            groups['exact:' + text].append(name)
        sig = signatures[name]
        if sig['actions']:
            key = v4.digest({k: sig[k] for k in ('actions', 'targets', 'restrictions', 'speed', 'secondary', 'replacement')})
            groups['signature:' + key].append(name)
    group_info = {}
    for key, members in groups.items():
        group_info[key] = (1 if key.startswith('exact:') else 2, 'identical_normalized_rules' if key.startswith('exact:') else 'heuristic_signature_match')
    for kind in ('roles', 'slots'):
        for label, raw in (curated or {}).get(kind, {}).items():
            key = f'curated:{kind}:{label}'
            groups[key] = sorted({normalize_card_name(n) for n in raw} & lookup.keys())
            # Broad role groups are proposals; only reviewed replacement slots tier 1.
            group_info[key] = (1 if kind == 'slots' else 2, key)
    for i, row in enumerate(replacements):
        if row['split'] != 'train':
            continue
        key = f'spellbook:{i}'
        groups[key] = sorted(set(row['alternatives']) & lookup.keys())
        group_info[key] = (1, 'train_only_spellbook_replacement')
    for members in groups.values():
        families.join(members)
    blocked_families = {families.find(n) for n in reserved if n in lookup}
    splits = {n: 'reserved_test' if families.find(n) in blocked_families else partition
              for n in lookup for partition in [v4.partition(families.find(n), seed)]}
    # External Spellbook positives retain TRAIN-only provenance. Entire overlapping
    # synthetic family is assigned train; blocked manual families take precedence.
    for key, members in groups.items():
        if key.startswith('spellbook:'):
            roots = {families.find(n) for n in members} - blocked_families
            for n in lookup:
                if families.find(n) in roots:
                    splits[n] = 'train'
    action_index = defaultdict(list)
    for name, sig in signatures.items():
        for action in sig['actions']:
            action_index[(splits[name], action)].append(name)
    rows = []
    negative_cache = {}
    for key, members in sorted(groups.items()):
        tier, rule = group_info[key]
        members = sorted(members)
        # Bounded deterministic candidates rather than quadratic broad-role expansion.
        for i, anchor in enumerate(members):
            if splits[anchor] == 'reserved_test':
                continue
            for positive in members[i + 1:i + 5]:
                if anchor not in negative_cache:
                    negatives = []
                    proposals = set()
                    for action in signatures[anchor]['actions']:
                        pool = sorted(action_index[(splits[anchor], action)])
                        generator = np.random.default_rng(int(v4.digest([seed, anchor, action])[:8], 16))
                        proposals.update(pool[:64])
                        proposals.update(pool[i] for i in generator.choice(len(pool), min(64, len(pool)), replace=False))
                    for negative in sorted(proposals):
                        if families.find(negative) == families.find(anchor):
                            continue
                        reasons = hard_negative_reasons(signatures[anchor], signatures[negative])
                        if reasons:
                            negatives.append({'card': negative, 'reasons': reasons, 'weight': .5})
                        if len(negatives) == 4:
                            break
                    negative_cache[anchor] = negatives
                negatives = negative_cache[anchor]
                rows.append(dict(anchor=anchor, positive=positive, negatives=negatives,
                                 tier=tier, confidence='high' if tier == 1 else 'proposal',
                                 weight=1. if tier == 1 else .5, task='card_to_card', derivation_rule=rule,
                                 family=families.find(anchor), split=splits[anchor],
                                 source_metadata={'oracle_ids': [lookup[n]['oracle_id'] for n in (anchor, positive)],
                                                  'group': key, 'signature': signatures[anchor],
                                                  'spellbook': replacements[int(key.split(':')[1])] if key.startswith('spellbook:') else None}))
    for name, card in sorted(lookup.items()):
        if splits[name] != 'reserved_test' and signatures[name]['actions']:
            rows.append(dict(anchor=name, positive=name, description=description(card), negatives=[],
                             tier=3, confidence='deterministic_description', weight=.25,
                             task='description_to_card', derivation_rule='metadata_and_exact_rules_template',
                             family=families.find(name), split=splits[name],
                             source_metadata={'oracle_ids': [card['oracle_id']], 'signature': signatures[name]}))
    return rows, splits


def prepare_oracle(source, output, deck_vocab, relations, benchmark, replacements=None, seed=42):
    inputs = [source, deck_vocab, relations, benchmark] + ([replacements] if replacements else [])
    with v4.stage(output, {'stage': 'oracle', 'seed': seed, 'views': VIEWS}, inputs) as dest:
        by_id = defaultdict(list)
        for card in iter_oracle_cards(Path(source)):
            if not card.get('oracle_id'):
                raise ValueError('Oracle identity required; cannot deduplicate by printing/name')
            by_id[card['oracle_id']].append(dict(card))
        cards = []
        conflicts = []
        canonical_ids = defaultdict(set)
        for oid, variants in sorted(by_id.items()):
            if len({serialize(c) for c in variants}) > 1:
                conflicts.append(oid)
                continue
            card = min(variants, key=lambda c: c.get('id', ''))
            card['canonical'] = normalize_card_name(card['name'])
            canonical_ids[card['canonical']].add(oid)
            cards.append(card)
        ambiguous = {n: sorted(ids) for n, ids in canonical_ids.items() if len(ids) > 1}
        cards = sorted((c for c in cards if c['canonical'] not in ambiguous and c['canonical']), key=lambda c: c['canonical'])
        declared = v4.read_json(benchmark)
        reserved = {normalize_card_name(n) for r in declared['records'] for n in
                    [r['anchor'], *r['positives'], *r.get('hard_negatives', [])]}
        replacements_data = v4.read_json(replacements) if replacements else []
        synthetic, splits = synthetic_dataset(cards, v4.read_json(relations), reserved, replacements_data, seed)
        corpus = [{'canonical': c['canonical'], 'oracle_id': c['oracle_id'], 'split':
                   'validation' if v4.partition(normalized_rules(c) or c['oracle_id'], seed) != 'train' else 'train',
                   'views': {view: serialize(c, view) for view in VIEWS}} for c in cards]
        deck = set(Path(deck_vocab).read_text(encoding='utf-8').splitlines())
        vocab = [c['canonical'] for c in cards]
        v4.write_json(dest / 'cards.json', cards)
        v4.write_json(dest / 'corpus.json', corpus)
        v4.write_json(dest / 'tokens.json', discover_tokens(cards))
        v4.write_json(dest / 'synthetic.json', synthetic)
        v4.write_json(dest / 'semantic_splits.json', splits)
        v4.write_json(dest / 'reserved_benchmark.json', declared)
        v4.write_json(dest / 'coverage.json', {'oracle_identities': len(by_id), 'usable_oracle_cards': len(vocab),
            'deck_vocabulary': len(deck), 'deck_overlap': len(deck & set(vocab)), 'deck_unseen': sorted(set(vocab) - deck),
            'deck_missing_text': sorted(deck - set(vocab)), 'ambiguous_keys': ambiguous, 'conflicting_identities': conflicts,
            'corpus_hash': v4.digest(corpus), 'synthetic_hash': v4.digest(synthetic),
            'split_counts': dict(Counter(splits.values())), 'synthetic_counts': dict(Counter(f"{r['split']}:tier{r['tier']}" for r in synthetic)),
            'character_lengths': {'maximum': max(map(lambda r: len(r['views']['structured_no_name']), corpus), default=0)},
            'manual_test_status': declared['review_status'],
            'mlm_split': 'normalized rules family hash; all serialization views stay together; no relation/probe labels'})
    return Path(output)


def pool_hidden(hidden, mask, method='mean'):
    """NumPy reference for masking tests; Torch training uses the same formula."""
    hidden, mask = np.asarray(hidden), np.asarray(mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.sum(axis=1).all():
        raise ValueError('Expected nonempty batch-aligned attention masks')
    if method == 'mean':
        return (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    if method == 'first':
        return hidden[np.arange(len(hidden)), mask.astype(bool).argmax(axis=1)]
    raise ValueError('Pooling must be mean or first')


def torch_pool(hidden, mask, method='mean'):
    if method == 'mean':
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    if method == 'first':
        return hidden[:, 0]  # tokenizer uses right padding and ModernBERT BOS/CLS
    raise ValueError('Pooling must be mean or first')


def download_model(output, revision='main'):
    from huggingface_hub import snapshot_download
    files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    with v4.stage(output, {'stage': 'download_model', 'model': MODEL, 'revision': revision, 'files': files}) as dest:
        snapshot_download(MODEL, revision=revision, local_dir=dest / 'model', allow_patterns=files)


def load_model(model_path, tokens=None):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'right'
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True,
                                               attn_implementation='sdpa', reference_compile=False)
    if tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': tokens['fields']})
        tokenizer.add_tokens(tokens['symbols'])
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def model_source(model_path, prepared=None):
    """Keep training lineage, and reject mixing a tuned model with another split."""
    model_path = Path(model_path)
    parent = model_path.parent
    if not (parent / 'complete.json').exists():
        return model_path, None
    provenance = v4.validate_stage(parent)
    if prepared is not None and provenance['config'].get('stage') in ('mlm', 'contrastive'):
        expected = provenance['inputs'].get(str((Path(prepared) / 'provenance.json').resolve()))
        if expected != v4.sha256_file(Path(prepared) / 'provenance.json'):
            raise ValueError('Model training corpus/split lineage differs from --prepared')
    return parent, provenance


def mlm_documents(corpus, synthetic=(), fraction=.2, semantic_records=False, seed=42):
    """Keep Oracle views; mix structured hypothetical records only into TRAIN.

    Validation retains each source's full held-out population, without oversampling.
    The returned document list records every sampled synthetic ID and repetition.
    """
    if not 0 <= fraction < 1:
        raise ValueError('Synthetic fraction must be in [0, 1)')
    views = VIEWS if semantic_records else VIEWS[:2]
    documents = {split: [{'id': r['oracle_id'], 'canonical': r['canonical'], 'source': 'oracle',
                         'split': split, 'view': view, 'text': r['views'][view]}
                        for r in corpus if r['split'] == split for view in views]
                 for split in ('train', 'validation')}
    if synthetic:
        hypothetical = {split: [{'id': r['synthetic_id'], 'canonical': r['canonical'],
                                  'source': 'synthetic_design', 'family': r['family'],
                                  'split': split, 'view': 'structured_no_name',
                                  'text': r['views']['structured_no_name']}
                                 for r in synthetic if r['split'] == split]
                        for split in documents}
        if any(not rows for rows in hypothetical.values()):
            raise ValueError('Synthetic design train/validation split is empty')
        if fraction == 0:
            raise ValueError('Supplying synthetic cards requires a positive mixing fraction')
        count = round(len(documents['train']) * fraction / (1 - fraction))
        rng = np.random.default_rng(seed)
        families = defaultdict(list)
        for row in hypothetical['train']:
            families[row['family']].append(row)
        # Equal family exposure prevents large Cartesian composition grids from
        # overwhelming simple effects. Parameter instances are sampled within it.
        for family in rng.choice(sorted(families), count):
            pool = families[family]
            documents['train'].append(pool[int(rng.integers(len(pool)))])
        documents['validation'].extend(hypothetical['validation'])
    return documents


def train_mlm(prepared, model_path, output, seed=42, epochs=1, semantic_records=False,
              max_length=256, batch_size=8, synthetic_cards=None, synthetic_fraction=.2):
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
    v4.validate_stage(prepared)
    if not 1 <= epochs <= 3:
        raise ValueError('Initial MLM configurations support 1-3 epochs')
    config = dict(stage='mlm', seed=seed, epochs=epochs, semantic_records=semantic_records,
                  max_length=max_length, batch_size=batch_size, learning_rate=2e-5,
                  weight_decay=.01, warmup_ratio=.06, lr_scheduler_type='cosine', mlm_probability=.15,
                  synthetic_designs=bool(synthetic_cards), synthetic_train_fraction=synthetic_fraction if synthetic_cards else 0,
                  synthetic_sampling='uniform family, then uniform parameter instance, with replacement',
                  validation_mask_seed=seed + 1000)
    corpus = v4.read_json(Path(prepared) / 'corpus.json')
    hypothetical = []
    tokens = v4.read_json(Path(prepared) / 'tokens.json')
    if synthetic_cards:
        provenance = v4.validate_stage(synthetic_cards)
        if provenance['config'].get('stage') != 'synthetic_cards':
            raise ValueError('Expected a prepared hypothetical-card stage, not synthetic relation pairs')
        coverage = v4.read_json(Path(synthetic_cards) / 'coverage.json')
        if coverage['source_oracle_corpus_hash'] != v4.digest(corpus):
            raise ValueError('Synthetic cards were prepared against another Oracle corpus/split')
        hypothetical = v4.read_json(Path(synthetic_cards) / 'corpus.json')
        synthetic_tokens = v4.read_json(Path(synthetic_cards) / 'tokens.json')
        tokens = {field: sorted(set(tokens[field]) | set(synthetic_tokens[field])) for field in tokens}
    documents = mlm_documents(corpus, hypothetical, synthetic_fraction, semantic_records, seed)
    with v4.stage(output, config, [prepared, model_path] + ([synthetic_cards] if synthetic_cards else [])) as dest:
        set_seed(seed)
        tokenizer, model = load_model(model_path, tokens)
        datasets = {split: [tokenizer(r['text'], truncation=True, max_length=max_length,
                                     return_special_tokens_mask=True) for r in rows]
                    for split, rows in documents.items()}
        if any(not rows for rows in datasets.values()):
            raise ValueError('MLM train/validation split is empty')
        lengths = [len(tokenizer(r['text'])['input_ids']) for rows in documents.values() for r in rows]
        v4.write_json(dest / 'token_lengths.json', {'max': max(lengths), 'p95': float(np.quantile(lengths, .95)),
                                                  'truncated_records': sum(n > max_length for n in lengths)})
        v4.write_json(dest / 'cards_used.json', [{'id': r.get('oracle_id', r.get('synthetic_id')), 'canonical': r['canonical'],
                                               'source': 'oracle' if 'oracle_id' in r else 'synthetic_design', 'split': r['split']}
                                              for r in corpus + hypothetical])
        v4.write_json(dest / 'mlm_documents.json', {split: [{k: v for k, v in r.items() if k != 'text'} for r in rows]
                                                 for split, rows in documents.items()})
        v4.write_json(dest / 'mixing.json', {split: dict(Counter(r['source'] for r in rows)) for split, rows in documents.items()})
        v4.write_json(dest / 'synthetic_family_exposure.json', dict(Counter(r['family'] for r in documents['train']
                                                                         if r['source'] == 'synthetic_design')))
        args = TrainingArguments(output_dir=str(dest / 'trainer'), num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
            learning_rate=2e-5, weight_decay=.01, warmup_ratio=.06, lr_scheduler_type='cosine',
            eval_strategy='epoch', save_strategy='no', logging_steps=50, seed=seed, data_seed=seed,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(), report_to=[],
            dataloader_num_workers=0, full_determinism=True)
        v4.write_json(dest / 'training_arguments.json', args.to_dict())
        trainer = Trainer(model=model, args=args, train_dataset=datasets['train'], eval_dataset=datasets['validation'],
                          data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=.15))
        def evaluate_sources():
            set_seed(seed + 1000)
            overall = trainer.evaluate()
            if not hypothetical:
                return overall, {'oracle': overall}
            sources = {}
            for source in sorted({r['source'] for r in documents['validation']}):
                set_seed(seed + 1000)
                data = [data for row, data in zip(documents['validation'], datasets['validation']) if row['source'] == source]
                sources[source] = trainer.evaluate(eval_dataset=data)
            return overall, sources
        before, sources_before = evaluate_sources()
        set_seed(seed)
        training = trainer.train()
        after, sources_after = evaluate_sources()
        trainer.save_model(str(dest / 'model'))
        tokenizer.save_pretrained(dest / 'model')
        v4.write_json(dest / 'losses.json', {'validation_before': before, 'training': training.metrics,
                                          'validation_after': after, 'history': trainer.state.log_history,
                                          'validation_by_source_before': sources_before,
                                          'validation_by_source_after': sources_after,
                                          'note': 'Dynamic masking loss; not a card-embedding metric'})


def encode_texts(tokenizer, encoder, texts, device, max_length=256, pooling='mean'):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden = encoder(**inputs).last_hidden_state
    return torch_pool(hidden, inputs['attention_mask'], pooling)


def contrastive_loss(anchor, positive, examples, hard_vectors, temperature=.07):
    """Multi-positive in-batch InfoNCE plus weighted explicit hard negatives."""
    import torch
    from torch.nn import functional as F
    a, p = F.normalize(anchor, dim=1), F.normalize(positive, dim=1)
    scores = a @ p.T / temperature
    family = [r['family'] for r in examples]
    positives = torch.tensor([[family[i] == family[j] for j in range(len(family))]
                              for i in range(len(family))], device=a.device)
    numerator = torch.logsumexp(scores.masked_fill(~positives, -torch.inf), dim=1)
    losses = []
    for i, row in enumerate(examples):
        candidates = [scores[i]]
        if hard_vectors[i] is not None:
            weights = torch.tensor([n['weight'] for n in row['negatives']], device=a.device)
            hard = F.normalize(hard_vectors[i], dim=1) @ a[i] / temperature
            candidates.append(hard + weights.log())
        losses.append(torch.logsumexp(torch.cat(candidates), dim=0) - numerator[i])
    weights = torch.tensor([r['weight'] for r in examples], device=a.device)
    return (torch.stack(losses) * weights).sum() / weights.sum()


def train_contrastive(prepared, model_path, output, seed=42, epochs=1, tiers=(1,),
                      descriptions=False, batch_size=16, max_length=256):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from transformers.trainer_utils import enable_full_determinism
    v4.validate_stage(prepared)
    model_input, lineage = model_source(model_path, prepared)
    if lineage is not None and lineage['config'].get('stage') != 'mlm':
        raise ValueError('Contrastive stage requires a separately evaluated MLM checkpoint')
    config = dict(stage='contrastive', seed=seed, epochs=epochs, tiers=list(tiers), descriptions=descriptions,
                  batch_size=batch_size, max_length=max_length, primary_pooling='mean',
                  temperature=.07, learning_rate=2e-5, weight_decay=.01, warmup_ratio=.06)
    with v4.stage(output, config, [prepared, model_input]) as dest:
        enable_full_determinism(seed)
        tokenizer, model = load_model(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        encoder = model.base_model
        cards = {c['canonical']: serialize(c) for c in v4.read_json(Path(prepared) / 'cards.json')}
        examples = [r for r in v4.read_json(Path(prepared) / 'synthetic.json')
                    if r['split'] == 'train' and ((r['tier'] in tiers and r['task'] == 'card_to_card')
                                                or (descriptions and r['task'] == 'description_to_card'))]
        if not examples:
            raise ValueError('No train-only synthetic examples for requested ablation')
        v4.write_json(dest / 'training_examples.json', examples)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-5, weight_decay=.01)
        steps = epochs * ((len(examples) + batch_size - 1) // batch_size)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(.06 * steps), steps)
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(epochs):
            encoder.train()
            losses = []
            order = [examples[i] for i in rng.permutation(len(examples))]
            for start in range(0, len(examples), batch_size):
                batch = order[start:start + batch_size]
                optimizer.zero_grad()
                a = encode_texts(tokenizer, encoder, [r.get('description', cards[r['anchor']]) for r in batch], device, max_length)
                p = encode_texts(tokenizer, encoder, [cards[r['positive']] for r in batch], device, max_length)
                hard = [encode_texts(tokenizer, encoder, [cards[n['card']] for n in r['negatives']], device, max_length)
                        if r['negatives'] else None for r in batch]
                loss = contrastive_loss(a, p, batch, hard)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            history.append({'epoch': epoch + 1, 'train_infonce': float(np.mean(losses))})
        model.save_pretrained(dest / 'model')
        tokenizer.save_pretrained(dest / 'model')
        v4.write_json(dest / 'losses.json', history)


def export_vectors(prepared, model_path, output, label, deck_vocab, view='structured_no_name',
                   batch_size=16, max_length=256):
    import torch
    v4.validate_stage(prepared)
    model_input, lineage = model_source(model_path, prepared)
    with v4.stage(output, dict(stage='semantic_export', label=label, view=view, max_length=max_length,
                              primary_pooling='mean', alternative='first token / ModernBERT BOS-CLS, diagnostic only'),
                  [prepared, model_input, deck_vocab]) as dest:
        v4.write_json(dest / 'model_lineage.json', lineage or {'source': str(Path(model_path).resolve()), 'training': 'external/local checkpoint; configuration in model/config.json'})
        tokenizer, model = load_model(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = model.base_model.to(device).eval()
        cards = v4.read_json(Path(prepared) / 'cards.json')
        matrices = {method: [] for method in ('mean', 'first')}
        descriptions = []
        truncated = 0
        with torch.inference_mode():
            for start in range(0, len(cards), batch_size):
                texts = [serialize(c, view) for c in cards[start:start + batch_size]]
                truncated += sum(len(tokenizer(t)['input_ids']) > max_length for t in texts)
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = encoder(**inputs).last_hidden_state
                for method in matrices:
                    matrices[method].append(torch_pool(hidden, inputs['attention_mask'], method).float().cpu().numpy())
                descriptions.append(encode_texts(tokenizer, encoder,
                    [description(c) for c in cards[start:start + batch_size]], device, max_length).float().cpu().numpy())
        vocab = [c['canonical'] for c in cards]
        (dest / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
        deck_names = Path(deck_vocab).read_text(encoding='utf-8').splitlines()
        from .static_multiview import align_view
        for method, batches in matrices.items():
            matrix = np.concatenate(batches).astype(np.float32)
            np.save(dest / f'{method}.npy', matrix, allow_pickle=False)
            aligned, mask = align_view(deck_names, vocab, matrix)
            np.save(dest / f'deck_{method}.npy', aligned, allow_pickle=False)
        np.save(dest / 'deck_mask.npy', mask, allow_pickle=False)
        np.save(dest / 'descriptions.npy', np.concatenate(descriptions).astype(np.float32), allow_pickle=False)
        v4.write_json(dest / 'coverage.json', {'full_oracle': len(vocab), 'deck_overlap': int(mask.sum()),
            'deck_vocab_hash': v4.digest(deck_names), 'deck_unseen': sorted(set(vocab) - set(deck_names)),
            'truncated_cards': truncated, 'description_source': 'deterministic metadata/exact Oracle rules template'})


def evaluate_semantic(prepared, vectors, output):
    """Validation/test reporting without using scores to pick pooling or supervision."""
    from .static_multiview import retrieval_metrics
    v4.validate_stage(prepared)
    v4.validate_stage(vectors)
    with v4.stage(output, {'stage': 'semantic_evaluation', 'primary': 'mean'}, [prepared, vectors]) as dest:
        names = (Path(vectors) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
        lookup = {n: i for i, n in enumerate(names)}
        rows = v4.read_json(Path(prepared) / 'synthetic.json')
        declared = v4.read_json(Path(prepared) / 'reserved_benchmark.json')
        results = []
        descriptions = unit_rows(np.load(Path(vectors) / 'descriptions.npy', allow_pickle=False))
        for method in ('mean', 'first'):
            matrix = unit_rows(np.load(Path(vectors) / f'{method}.npy', allow_pickle=False))
            for row in rows:
                if row['split'] not in ('validation', 'test'):
                    continue
                anchor, target = lookup[row['anchor']], lookup[row['positive']]
                description_task = row['task'] == 'description_to_card'
                query = descriptions[anchor] if description_task else matrix[anchor]
                results.append({'split': row['split'], 'tier': row['tier'], 'pooling': method,
                                'task': row['task'], 'anchor': row['anchor'], 'target': row['positive'],
                                'hard_negative_margins': {n['card']: float(matrix[target] @ query - matrix[lookup[n['card']]] @ query)
                                                         for n in row['negatives']},
                                **retrieval_metrics(matrix @ query, [] if description_task else [anchor], [target], names)[0]})
            if declared['review_status'] == 'manually_reviewed':
                for row in declared['records']:
                    if row['anchor'] in lookup and all(n in lookup for n in row['positives']):
                        a = lookup[row['anchor']]
                        results.append({'split': 'untouched_manual_test', 'pooling': method, 'anchor': row['anchor'],
                                        **retrieval_metrics(matrix @ matrix[a], [a], [lookup[n] for n in row['positives']], names)[0]})
        if not results:
            results_unavailable = [{'task': 'synthetic_retrieval', 'reason': 'no held-out synthetic examples'}]
        else:
            results_unavailable = []
        v4.write_json(dest / 'retrieval.json', results)
        v4.write_json(dest / 'unavailable.json', results_unavailable + ([{'task': 'untouched_manual_test', 'reason': declared['review_status']}]
                      if declared['review_status'] != 'manually_reviewed' else []))
