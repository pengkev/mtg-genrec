import copy
import pytest
import torch

from mtgdeck.data import build_vocabulary, normalize_deck_record
from mtgdeck.genrec_training import (canonical_format, canonical_model_deck, FormatCandidateIndex,
    split_joint_corpora, make_model, train_stage, evaluate_model)
from mtgdeck.legality import OracleCatalog


def card(name, oid, **extra):
    return dict(name=name, oracle_id=oid, type_line='Creature',
                legalities={'modern':'legal', 'legacy':'legal', 'commander':'legal'}, **extra)


def fixture():
    catalog = OracleCatalog([
        card('Leader', 'leader', color_identity=['U']),
        card('Island', 'island', color_identity=['U']),
        card('Red Spell', 'red', color_identity=['R']),
    ])
    catalog.resolve('Leader')['type_line'] = 'Legendary Creature'
    catalog.resolve('Island')['type_line'] = 'Basic Land'
    record = normalize_deck_record({'id':'x', 'format':'modern', 'mainboard':[
        {'n':'Island','q':56}, {'n':'Red Spell','q':4}]}, 'moxfield')
    return catalog, record


def test_format_aliases_and_distinct_legality_masks():
    assert [canonical_format(x) for x in ('edh','cEDH','commander')] == ['commander']*3
    assert canonical_format('duel-commander') != 'commander'
    catalog, _ = fixture()
    catalog.resolve('Red Spell')['legalities']['modern'] = 'banned'
    vocab = {'<PAD>':0, '<UNK>':1, 'oid:leader':2, 'oid:island':3, 'oid:red':4}
    deck = {'commanders':[{'name':'Leader'}]}
    assert not FormatCandidateIndex(catalog, vocab, 'modern').allowed_mask({})[4]
    assert FormatCandidateIndex(catalog, vocab, 'legacy').allowed_mask({})[4]
    assert not FormatCandidateIndex(catalog, vocab, 'edh').allowed_mask(deck)[4]
    assert not FormatCandidateIndex(catalog, vocab, 'legacy').allowed_mask({})[:2].any()


def test_constructed_validation_merges_identities_and_preserves_quantities():
    catalog, record = fixture()
    deck = canonical_model_deck(record, catalog, 'modern')
    assert deck['commanders'] == []
    assert deck['mainboard'][0]['quantity'] == 56
    assert deck['mainboard'][0]['name'] == 'oid:island'
    assert record['mainboard'][0]['name'] == 'Island'
    record['sideboard'] = [{'name':'Red Spell','quantity':1}]
    with pytest.raises(ValueError, match='copy_limit'):
        canonical_model_deck(record, catalog, 'modern')
    record['sideboard'] = []
    record['mainboard'][0]['quantity'] = 55
    with pytest.raises(ValueError, match='below_60'):
        canonical_model_deck(record, catalog, 'modern')


def test_commander_alias_validation():
    catalog, record = fixture()
    record.update(format='cedh', commanders=[{'name':'Leader','quantity':1}],
                  mainboard=[{'name':'Island','quantity':98}, {'name':'Red Spell','quantity':1}])
    catalog.resolve('Red Spell')['color_identity'] = ['U']
    deck = canonical_model_deck(record, catalog, 'commander')
    assert deck['format'] == 'commander' and deck['commanders'][0]['name'] == 'oid:leader'


def rows(prefix, n=30):
    return [{'deck_id':f'{prefix}{i}', 'commanders':[], 'mainboard':[
        {'name':f'oid:{prefix}{i}_{j}', 'quantity':1} for j in range(20)]} for i in range(n)]


def test_joint_split_keeps_cross_tier_near_duplicates_together():
    base, premium = rows('base'), rows('premium')
    # Same model inputs but different quantities/sideboards still cannot leak.
    duplicate = copy.deepcopy(base[0]); duplicate['deck_id'] = 'premium_copy'
    premium.append(duplicate)
    near = copy.deepcopy(base[1]); near['deck_id'] = 'premium_near'
    near['mainboard'][-1]['name'] = 'oid:replacement'
    premium.append(near)
    splits, audit = split_joint_corpora(base, premium)
    assert audit['deduplicated'] == 1
    membership = {row['deck_id']:row for row in audit['membership']}
    assert 'base0' not in membership
    assert membership['base1']['split'] == membership['premium_near']['split']
    assert membership['base1']['group'] == membership['premium_near']['group']
    assert all(rows for tiers in splits.values() for rows in tiers.values())
    assert split_joint_corpora(base, premium)[1] == audit


def test_changed_source_version_cannot_cross_tier_splits():
    base, premium = rows('base'), rows('premium')
    premium[0]['deck_id'] = base[0]['deck_id']
    _, audit = split_joint_corpora(base, premium)
    versions = [row for row in audit['membership'] if row['deck_id'] == 'base0']
    assert len(versions) == 2
    assert len({row['group'] for row in versions}) == len({row['split'] for row in versions}) == 1


def test_demo_discovers_all_format_checkpoints(tmp_path):
    from mtgdeck.inference import available_checkpoints
    for fmt in ('commander', 'modern', 'legacy'):
        (tmp_path / f'attention_oracleid_v2_{fmt}_static_premium_384.pt').touch()
    paths = available_checkpoints(tmp_path)
    assert len(paths) == 3
    assert {path.stem.split('_')[3] for path in paths} == {'commander', 'modern', 'legacy'}


@pytest.mark.parametrize('fmt', ['commander','modern','legacy'])
def test_both_training_stages_and_matched_evaluation(tmp_path, fmt):
    torch.set_num_threads(1)
    catalog, _ = fixture()
    catalog.resolve('Red Spell')['color_identity'] = ['U']
    # Many legal alternatives provide nonzero recommendation gradients.
    cards = [card(f'Spell {i}', f'spell{i}', color_identity=['U']) for i in range(8)]
    catalog = OracleCatalog([catalog.resolve(name) for name in ('Leader','Island','Red Spell')] + cards)
    decks = []
    for i in range(8):
        decks.append({'deck_id':str(i), 'commanders':([{'name':'oid:leader','oracle_id':'leader','quantity':1}] if fmt=='commander' else []),
                      'mainboard':[{'name':f'oid:spell{j}', 'oracle_id':f'spell{j}', 'quantity':1} for j in (i, (i+1)%8, (i+2)%8)]})
    vocab = build_vocabulary(decks)
    weights = torch.randn(len(vocab), 384); weights[:2] = 0
    splits = {tier:{name:decks for name in ('train','validation','test')} for tier in ('base','premium')}
    prepared = {'format':fmt, 'vocab':vocab, 'weights':weights, 'splits':splits,
                'candidates':FormatCandidateIndex(catalog,vocab,fmt), 'audit':{'embeddings':{}}}
    cfg = dict(seed=42, device='cpu', model_dim=16, heads=4, blocks=1, latent_dim=8,
               pool_queries=2, decoder_queries=2, batch_size=4, eval_batch_size=4,
               train_mask_ratios=(.3,), eval_mask_ratio=.3, eval_repeats=1, kl_beta=.01,
               base=dict(epochs=1,learning_rate=2e-4,embedding_learning_rate=2e-6,kl_warmup_steps=5),
               premium=dict(epochs=1,learning_rate=2e-5,embedding_learning_rate=2e-7,kl_warmup_steps=2))
    model = make_model(prepared,cfg)
    base = tmp_path/'base.pt'; premium = tmp_path/'premium.pt'
    train_stage(model,prepared,cfg,'base',base)
    assert not torch.equal(model.card_embedding.weight,weights)
    # Stage two explicitly reloads the parent, not arbitrary in-memory weights.
    expected = evaluate_model(model,decks,prepared,cfg)
    with torch.no_grad(): model.card_embedding.weight.add_(10)
    result = train_stage(model,prepared,cfg,'premium',premium,parent_checkpoint=base)
    assert result['history'][0]['validation'] == expected
    saved = torch.load(premium,weights_only=True)
    assert saved['config']['format']==fmt and saved['parent_checkpoint']==str(base)
    assert saved['config']['premium']['embedding_learning_rate'] == 2e-7
    assert evaluate_model(model,decks,prepared,cfg)['tasks'] == 8
    assert torch.equal(model.card_embedding.weight[:2],weights[:2])
    with pytest.raises(FileExistsError): train_stage(model,prepared,cfg,'base',base)
