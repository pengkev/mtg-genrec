"""Offline contracts for v4 identities, supervision isolation and portable views."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mtgdeck import static_semantic as semantic
from mtgdeck import static_spellbook as spellbook
from mtgdeck import static_multiview as multiview
from mtgdeck import static_v4 as v4
from mtgdeck.static_export import fuse_sources, write_checkpoint


def card(name, text='Draw a card.', **extra):
    return dict(name=name, canonical=name.lower(), oracle_id='oid:' + name.lower(),
                oracle_text=text, mana_cost='{2/W}{U/P}{X}', type_line='Instant', cmc=2, **extra)


def variant(ident, members, family='1', **extra):
    return dict(id=ident, of=[{'id': family}], includes=[], status='OK',
                uses=[{'card': {'name': n, 'oracleId': 'oid:' + n.lower()}, 'quantity': 1} for n in members],
                requires=[], produces=[{'feature': {'id': 1, 'name': 'Draw cards'}, 'quantity': 1}],
                legalities={'commander': True}, description='Steps', **extra)


def test_serialization_faces_fields_and_name_identity():
    c = card('Tester', 'Tester taps: Add {E}.', flavor_text='Never train this', power='1', toughness='2', loyalty='3', defense='4')
    text = semantic.serialize(c)
    assert '[SELF] taps' in text and 'Tester' not in text
    assert '[PT] 1/2' in text and '[LOYALTY] 3' in text and '[DEFENSE] 4' in text
    assert 'Never train' not in text and '[TYPE]' not in semantic.serialize(c, 'rules_only')
    c['card_faces'] = [dict(name='Front', oracle_text='Front draws.', type_line='Creature'),
                       dict(name='Back', oracle_text='Back adds {Q}.', type_line='Land')]
    assert '[FACE]' in semantic.serialize(c) and '[SELF] adds {Q}' in semantic.serialize(c)
    with pytest.raises(ValueError):
        semantic.serialize(c, 'name_only')


def test_symbols_and_auditable_signature():
    c = card('Test', 'Counter target noncreature spell. Draw a card. {T} {Q} {S} {E} {CHAOS}')
    tokens = semantic.discover_tokens([c])
    assert set(tokens['symbols']) == {'{2/W}', '{U/P}', '{X}', '{T}', '{Q}', '{S}', '{E}', '{CHAOS}'}
    sig = semantic.effect_signature(c)
    assert sig['actions'] == ['counter', 'draw']
    assert sig['evidence']['action:counter'] == 'counter target noncreature spell'
    assert 'noncreature' in sig['restrictions']
    assert '[ACTION]' in semantic.serialize(c, 'normalized_semantic')


def test_semantic_family_splits_and_reserved_transitivity():
    cards = [card('A'), card('B'), card('C', 'Destroy target creature.'), card('D', 'Exile target creature.')]
    curated = {'slots': {'overlap': ['B', 'C']}}
    first, splits = semantic.synthetic_dataset(cards, curated, reserved={'a'})
    second, other = semantic.synthetic_dataset(list(reversed(cards)), curated, reserved={'a'})
    assert first == second and splits == other
    assert all(splits[n] == 'reserved_test' for n in ('a', 'b', 'c'))
    assert not any({'a', 'b', 'c'} & {r['anchor'], r['positive']} for r in first)
    assert all(splits[n['card']] == r['split'] for r in first for n in r['negatives'])


def test_spellbook_train_replacements_never_use_test_slots():
    cards = [card('A'), card('B', 'Discard a card.')]
    slots = [{'split': 'test', 'alternatives': ['a', 'b']}, {'split': 'train', 'alternatives': ['a', 'b']}]
    rows, splits = semantic.synthetic_dataset(cards, replacements=slots)
    relations = [r for r in rows if r['derivation_rule'] == 'train_only_spellbook_replacement']
    assert relations and all(r['split'] == 'train' for r in relations)
    assert all(r['source_metadata']['spellbook']['split'] == 'train' for r in relations)


def test_pooling_masks_padding_and_first_valid():
    hidden = np.array([[[100, 100], [1, 3], [3, 5]], [[2, 2], [4, 4], [999, 999]]])
    mask = np.array([[0, 1, 1], [1, 1, 0]])
    np.testing.assert_array_equal(semantic.pool_hidden(hidden, mask), [[2, 4], [3, 3]])
    np.testing.assert_array_equal(semantic.pool_hidden(hidden, mask, 'first'), [[1, 3], [2, 2]])
    with pytest.raises(ValueError):
        semantic.pool_hidden(hidden, np.zeros_like(mask))


def test_parser_preserves_actual_fields_rejects_identity_mismatch_and_templates():
    cards = [card('A'), card('B')]
    raw = variant('v', ['A', 'B'])
    raw['requires'] = [{'template': {'id': 4, 'name': 'Any creature'}}]
    rows, stats = spellbook.parse_variants({'variants': [raw]}, cards)
    assert rows[0]['raw'] == raw and rows[0]['templates'] == raw['requires']
    assert not rows[0]['eligible'] and stats['template_variants'] == 1
    raw['requires'] = []
    raw['uses'][0]['card']['oracleId'] = 'wrong identity'
    rows, _ = spellbook.parse_variants({'variants': [raw]}, cards)
    assert rows[0]['unresolved'][0]['reason'] == 'unknown_or_identity_mismatch'
    with pytest.raises(ValueError, match='Incomplete'):
        spellbook.variant_rows({'results': [raw], 'next': 'next page'})


def test_leave_one_out_and_conditional_replacement():
    cards = [card(n) for n in ['A', 'B', 'C', 'D']]
    raw = [variant('abc', ['A', 'B', 'C']), variant('abd', ['A', 'B', 'D'])]
    rows, _ = spellbook.parse_variants({'variants': raw}, cards)
    rows = spellbook.split_hyperedges(rows)
    examples = spellbook.leave_one_out(rows)
    assert len(examples) == 6
    assert all(r['target'] not in r['context'] and len(r['context']) == 2 for r in examples)
    slots = spellbook.replacement_slots(rows)
    slot = next(r for r in slots if r['shared_context'] == ['a', 'b'])
    assert slot['alternatives'] == ['c', 'd'] and slot['source_variant_ids'] == ['abc', 'abd']
    changed = copy.deepcopy(rows)
    changed[1]['outcomes'] = ['different']
    assert not spellbook.replacement_slots(changed)


def test_family_split_transitivity_exact_sets_and_overlap_audit():
    cards = [card(n) for n in ['A', 'B', 'C', 'D', 'E']]
    raws = [variant('a', ['A', 'B'], '1'), variant('b', ['A', 'C'], '1'),
            variant('c', ['A', 'C'], '2'), variant('d', ['D', 'E'], '3')]
    rows, _ = spellbook.parse_variants({'variants': raws}, cards)
    split = spellbook.split_hyperedges(rows)
    assert len({r['split_group'] for r in split[:3]}) == 1
    assert spellbook.split_hyperedges(list(reversed(rows))) == list(reversed(split))
    assert not spellbook.overlap_audit(split)['cross_split_overlap_counts']
    split[0]['split'] = 'train'
    split[1]['split'] = 'test'
    split[2]['split'] = 'validation'
    split.append({**split[0], 'id': 'superset', 'cards': ['a', 'b', 'c'], 'split': 'test'})
    audit = spellbook.overlap_audit(split)
    assert audit['cross_split_overlap_counts']['test:validation:exact_duplicate'] == 1
    assert audit['cross_split_overlap_counts']['test:train:subset_superset'] >= 1
    assert audit['shared_card_frequency']['a']['train'] == 1


def test_alignment_zero_missing_block_normalization_and_cold_start():
    source = np.array([[3, 4], [0, 2]], dtype=np.float32)
    aligned, mask = multiview.align_view(['b', 'absent', 'a'], ['a', 'b'], source, [True, False])
    np.testing.assert_array_equal(mask, [False, False, True])
    np.testing.assert_array_equal(aligned, [[0, 0], [0, 0], [3, 4]])
    fused = multiview.concatenate([aligned, aligned * 100])
    np.testing.assert_allclose(fused[2], [.6, .8, .6, .8])
    np.testing.assert_array_equal(source, [[3, 4], [0, 2]])
    np.testing.assert_array_equal(multiview.cold_start(aligned, [2]), np.zeros_like(aligned))
    with pytest.raises(ValueError):
        multiview.align_view(['a'], ['a', 'a'], source)


def test_rank_ties_and_geometry_unavailable():
    metrics, order = multiview.retrieval_metrics(np.zeros(4), [0], [2], ['d', 'c', 'b', 'a'])
    assert order.tolist() == [3, 2, 1] and metrics['mrr'] == .5 and metrics['recall@5'] == 1
    assert multiview.geometry(np.zeros((3, 2)), np.zeros((3, 4)))['linear_cka'] is None


def test_incidence_cosine_normalizes_the_aggregated_visible_set():
    pytest.importorskip('scipy')
    cards = [card(n) for n in 'ABCD']
    rows, _ = spellbook.parse_variants({'variants': [variant('abc', ['A', 'B', 'C']), variant('abd', ['A', 'B', 'D'])]}, cards)
    rows = [{**r, 'split': 'train'} for r in rows]
    score = spellbook.incidence_baselines(rows, ['a', 'b', 'c', 'd'])
    np.testing.assert_allclose(score([0, 1], 'incidence_cosine'), [1, 1, 2**-.5, 2**-.5])


def test_model_lineage_rejects_another_semantic_partition(tmp_path):
    first, other, trained = [tmp_path / n for n in ('first', 'other', 'trained')]
    for path, seed in [(first, 42), (other, 43)]:
        with v4.stage(path, {'stage': 'oracle', 'seed': seed}):
            pass
    with v4.stage(trained, {'stage': 'mlm', 'seed': 42}, [first]) as path:
        (path / 'model').mkdir()
    assert semantic.model_source(trained / 'model', first)[1]['config']['seed'] == 42
    with pytest.raises(ValueError, match='lineage differs'):
        semantic.model_source(trained / 'model', other)


def test_stage_write_once_hash_validation_and_protected_paths(tmp_path, monkeypatch):
    historical = tmp_path / 'history'
    historical.mkdir()
    monkeypatch.setattr(v4, 'historical_roots', lambda: [historical])
    with pytest.raises(ValueError, match='protected'):
        with v4.stage(historical / 'child', {}):
            pass
    dest = tmp_path / 'new'
    with v4.stage(dest, {'seed': 42}) as path:
        v4.write_json(path / 'result.json', {'value': 1})
    assert v4.validate_stage(dest)['config']['seed'] == 42
    with pytest.raises(FileExistsError):
        with v4.stage(dest, {}):
            pass
    (dest / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        v4.validate_stage(dest)


def test_portable_multiview_roundtrip_and_masks(tmp_path):
    deck = tmp_path / 'deck'
    arrays = [np.ones((2, 128), dtype=np.float32) * s for s in (1, 2, 3)]
    write_checkpoint(deck, ['a', 'b'], [(42, fuse_sources(*arrays))], {}, {'42': []})
    sem = tmp_path / 'semantic'
    with v4.stage(sem, {}) as path:
        (path / 'vocab.txt').write_text('b\nc\n')
        np.save(path / 'mean.npy', np.array([[3, 4], [1, 0]], dtype=np.float32))
    mech = tmp_path / 'mechanical'
    with v4.stage(mech, {}) as path:
        (path / 'vocab.txt').write_text('a\nb\n')
        np.save(path / 'vectors.npy', np.array([[1, 2], [8, 8]], dtype=np.float32))
        np.save(path / 'mask.npy', np.array([True, False]))
    output = tmp_path / 'v2'
    multiview.export(deck, {'mlm': sem}, mech, output)
    full = multiview.load_multiview_embeddings(output)
    assert full.vocab == ['a', 'b', 'c'] and full.embedding_matrix.shape == (3, 388)
    text = multiview.load_multiview_embeddings(output, 'text')
    np.testing.assert_array_equal(text.embedding_matrix, [[0, 0], [3, 4], [1, 0]])
    mechanical = multiview.load_multiview_embeddings(output, 'mechanical')
    np.testing.assert_array_equal(mechanical.embedding_matrix, [[1, 2], [0, 0], [0, 0]])
    assert full.metadata['ablation_status']['tier1'] == 'unavailable/pending'
    assert multiview.load_multiview_embeddings(output, 'sgns_ppmi').embedding_matrix.shape == (3, 256)
    delta = multiview.load_multiview_embeddings(output, 'difference').embedding_matrix
    np.testing.assert_array_equal(delta, np.zeros((3, 128)))
    combined = multiview.load_multiview_embeddings(output, 'sgns_ppmi_difference_text_mechanical')
    assert combined.embedding_matrix.shape == (3, 388)
    np.testing.assert_array_equal(combined.embedding_matrix[2, :384], 0)


def test_difference_uses_alignment_and_preserves_actual_discrepancy():
    rng = np.random.default_rng(82)
    sgns = rng.normal(size=(40, 4)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    ppmi = (sgns @ rotation).astype(np.float32)
    delta, fitted = multiview.aligned_difference(sgns, ppmi)
    np.testing.assert_array_equal(delta, np.zeros_like(sgns))
    np.testing.assert_allclose(fitted.T @ fitted, np.eye(4), atol=1e-6)
    ppmi[0] = ppmi[1]
    delta, fitted = multiview.aligned_difference(sgns, ppmi)
    assert np.linalg.norm(delta[0]) > .1
    np.testing.assert_allclose(delta, multiview.unit_rows(sgns) - multiview.unit_rows(ppmi) @ fitted, atol=1e-6)


def test_multiview_evaluation_keeps_mlm_roles_and_excludes_trained_combo_targets(tmp_path, monkeypatch):
    joblib = pytest.importorskip('joblib')
    from mtgdeck import static_relations as relations
    names = ['a', 'b', 'c', *[f'card{i}' for i in range(3, 60)]]
    deck = tmp_path / 'deck'
    rng = np.random.default_rng(42)
    arrays = [rng.normal(size=(len(names), 128)).astype(np.float32) for _ in range(3)]
    write_checkpoint(deck, names, [(42, fuse_sources(*arrays))], {}, {'42': []})
    sem = tmp_path / 'semantic'
    with v4.stage(sem, {}) as path:
        (path / 'vocab.txt').write_text('\n'.join(names))
        np.save(path / 'mean.npy', np.eye(len(names), dtype=np.float32))
        v4.write_json(path / 'model_lineage.json', {'config': {'stage': 'mlm'}})
    mech = tmp_path / 'mechanical'
    with v4.stage(mech, {}) as path:
        (path / 'vocab.txt').write_text('\n'.join(names))
        np.save(path / 'vectors.npy', np.eye(len(names), dtype=np.float32))
        np.save(path / 'mask.npy', np.ones(len(names), dtype=bool))
    checkpoint = tmp_path / 'export'
    multiview.export(deck, {'mlm': sem}, mech, checkpoint)
    oracle = tmp_path / 'oracle'
    with v4.stage(oracle, {}) as path:
        v4.write_json(path / 'cards.json', [card(n.upper()) for n in names])
        v4.write_json(path / 'synthetic.json', [{'family': 'role', 'split': 'train',
            'task': 'card_to_card', 'anchor': 'a', 'positive': 'b', 'tier': 1}])
    spell = tmp_path / 'spell'
    with v4.stage(spell, {}) as path:
        v4.write_json(path / 'variants.json', [{'eligible': True, 'split': 'train', 'cards': ['a', 'b']}])
        v4.write_json(path / 'completion.json', [{'variant_id': 'test', 'split': 'test', 'context': ['a'], 'target': 'c'}])
    inputs = tmp_path / 'artifacts/card2vec/static_v2'
    inputs.mkdir(parents=True)
    ids = np.arange(len(names))
    targets = {task: (ids, (ids % 2)[:, None], ['label']) for task in ('color', 'type', 'format', 'archetype')}
    targets['mana'] = (ids, (ids % 8).astype(float), ['mana'])
    joblib.dump({'names': names, 'counts': ids + 1, 'targets': targets,
                 'metadata': {n: {'type_line': 'Instant'} for n in names}}, inputs / 'evaluation_inputs.joblib')
    config = tmp_path / 'configs'
    config.mkdir()
    v4.write_json(config / 'card2vec_relations.json', {})
    records = [dict(id='role', query=['a'], targets=['b'], relation='similarity', source='test',
                    concept='role', status='covered'),
               dict(id='combo', query=['a'], targets=['b', 'c'], relation='complementarity',
                    source='test', concept='combo', status='covered')]
    monkeypatch.setattr(relations, 'build_benchmark', lambda *args: records)
    output = tmp_path / 'evaluation'
    multiview.evaluate(tmp_path, checkpoint, oracle, output, probes=True, spellbook=spell)
    queries = {r['id']: r for r in v4.read_json(output / 'queries.json')}
    assert queries['role']['targets'] == ['b']  # MLM never trained the role labels
    assert queries['combo']['targets'] == ['c']  # omit only the train-exposed target
    assert v4.read_json(output / 'leakage_exclusions.json')[0]['reason'] == 'spellbook_train_pair_overlap'
    import pandas as pd
    summary = pd.read_csv(output / 'relation_summary.csv')
    assert 'sgns_ppmi_difference_text_mechanical' in set(summary.representation)
    assert (output / 'spellbook_cosine_summary.csv').exists()
    probes = pd.read_csv(output / 'probe_summary.csv')
    assert set(probes.task) == {'color', 'type', 'format', 'mana', 'archetype'}
    assert (probes['count'] == 3).all() and np.isfinite(probes[['mean', 'std']]).all().all()
    assert set(probes.baseline) == {'learned', 'frequency', 'constant'}
    v4.validate_stage(output)


def test_set_model_order_invariance_distinct_heads_and_gradient():
    torch = pytest.importorskip('torch')
    model = spellbook.make_model(5, 8)
    ids = torch.tensor([[0, 1], [1, 0]])
    scores = model.score(ids, torch.ones_like(ids))
    torch.testing.assert_close(scores[0], scores[1])
    assert model.query['complementarity'] is not model.query['replacement']
    scores.sum().backward()
    assert model.cards.weight.grad is not None


def test_contrastive_loss_does_not_penalize_known_same_family():
    torch = pytest.importorskip('torch')
    vectors = torch.eye(2, requires_grad=True)
    examples = [{'family': 'shared', 'weight': 1, 'negatives': []}] * 2
    loss = semantic.contrastive_loss(vectors, vectors, examples, [None, None])
    assert float(loss.detach()) == pytest.approx(0)


def test_tiny_offline_modernbert_pipeline(tmp_path):
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'Draw': 5, 'cards': 6}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    cfg = transformers.ModernBertConfig(vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, local_attention=16, max_position_embeddings=256, pad_token_id=0, cls_token_id=2, sep_token_id=3,
        reference_compile=False)
    model_dir = tmp_path / 'model'
    transformers.ModernBertForMaskedLM(cfg).save_pretrained(model_dir)
    hf.save_pretrained(model_dir)
    cards = [card(f'Card{i:03}', f'Draw {i} cards.') for i in range(25)]
    source = tmp_path / 'oracle.json'
    v4.write_json(source, cards)
    vocab = tmp_path / 'deck_vocab.txt'
    vocab.write_text('\n'.join(c['canonical'] for c in cards[:20]))
    relations = tmp_path / 'relations.json'
    v4.write_json(relations, {'roles': {}, 'slots': {}})
    benchmark = tmp_path / 'benchmark.json'
    v4.write_json(benchmark, {'review_status': 'pending_human_review', 'records': []})
    prepared = tmp_path / 'oracle_prepared'
    semantic.prepare_oracle(source, prepared, vocab, relations, benchmark)
    from mtgdeck.static_synthetic_cards import prepare as prepare_designs
    designs = tmp_path / 'designs'
    prepare_designs(prepared, designs)
    trained = tmp_path / 'mlm'
    semantic.train_mlm(prepared, model_dir, trained, batch_size=32, synthetic_cards=designs)
    assert v4.read_json(trained / 'losses.json')['validation_after']['eval_loss'] > 0
    losses = v4.read_json(trained / 'losses.json')
    assert set(losses['validation_by_source_after']) == {'oracle', 'synthetic_design'}
    assert v4.read_json(trained / 'mixing.json')['train']['synthetic_design'] > 0
    vectors = tmp_path / 'vectors'
    semantic.export_vectors(prepared, trained / 'model', vectors, 'mlm', vocab, batch_size=32)
    assert np.load(vectors / 'mean.npy').shape == (25, 16)
    assert np.load(vectors / 'deck_mean.npy').shape == (20, 16)
    tuned = tmp_path / 'tuned'
    semantic.train_contrastive(prepared, trained / 'model', tuned, tiers=(1, 2), descriptions=True, batch_size=32)
    assert v4.read_json(tuned / 'training_examples.json')
    evaluation = tmp_path / 'evaluation'
    semantic.evaluate_semantic(prepared, vectors, evaluation)
    v4.validate_stage(evaluation)


def test_tiny_hyperedge_train_and_full_vocabulary_evaluation(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('scipy')
    names = [n.lower() for n in 'ABCDEFGH']
    cards = [card(n) for n in 'ABCDEFGH']
    ids = {}
    for i in range(1000):
        split = v4.partition('combo:' + str(i))
        ids.setdefault(split, str(i))
        if len(ids) == 3:
            break
    raws = [variant('abc', ['A', 'B', 'C'], ids['train']), variant('abd', ['A', 'B', 'D'], ids['train']),
            variant('efg', ['E', 'F', 'G'], ids['validation']), variant('ceh', ['C', 'E', 'H'], ids['test'])]
    rows, _ = spellbook.parse_variants({'variants': raws}, cards)
    rows = spellbook.split_hyperedges(rows)
    prepared = tmp_path / 'prepared'
    with v4.stage(prepared, {}) as path:
        (path / 'vocab.txt').write_text('\n'.join(names))
        v4.write_json(path / 'variants.json', rows)
        v4.write_json(path / 'completion.json', spellbook.leave_one_out(rows))
        v4.write_json(path / 'replacements.json', spellbook.replacement_slots(rows))
    trained = tmp_path / 'trained'
    spellbook.train(prepared, trained, dimension=8, epochs=1, batch_size=8, auxiliary=True)
    raw = np.load(trained / 'vectors.npy')
    mask = np.load(trained / 'mask.npy')
    assert np.all(raw[~mask] == 0) and mask.sum() == 4
    assert v4.read_json(trained / 'heads.json')['replacement'].startswith('trained')
    deck = tmp_path / 'deck'
    blocks = [np.random.default_rng(s).normal(size=(8, 128)).astype(np.float32) for s in range(3)]
    write_checkpoint(deck, names, [(42, fuse_sources(*blocks))], {}, {'42': []})
    dest = tmp_path / 'evaluation'
    spellbook.evaluate(prepared, trained, deck, dest)
    results = v4.read_json(dest / 'retrieval.json')
    assert len(results) == 3 * 7 and all(r['candidate_count'] == 6 for r in results)
    assert all(r['task'] == 'set_completion' for r in results)
    v4.validate_stage(dest)
