"""Explicit local stages for Static v4 semantic research."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtgdeck import static_semantic as semantic
from mtgdeck import static_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'prepare-designs', 'download-model', 'mlm', 'contrastive', 'export', 'evaluate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--oracle', type=Path, default=ROOT / 'data/oracle_cards.jsonl.gz')
    parser.add_argument('--prepared', type=Path, default=v4.DEFAULT_ROOT / 'oracle')
    parser.add_argument('--model', type=Path, default=v4.DEFAULT_ROOT / 'model_download_v2/model')
    parser.add_argument('--vectors', type=Path, default=v4.DEFAULT_ROOT / 'semantic_mlm')
    parser.add_argument('--deck-vocab', type=Path, default=ROOT / 'artifacts/card2vec/static_export_v1/vocab.txt')
    parser.add_argument('--relations', type=Path, default=ROOT / 'configs/card2vec_relations.json')
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'configs/card2vec_semantic_v4_test.json')
    parser.add_argument('--replacements', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--semantic-records', action='store_true')
    parser.add_argument('--synthetic-cards', type=Path, help='Prepared hypothetical cards for MLM; distinct from relation-pair supervision')
    parser.add_argument('--synthetic-fraction', type=float, default=.2, help='Fraction of TRAIN documents from hypothetical cards')
    parser.add_argument('--design-policy', type=Path, default=ROOT / 'configs/card2vec_synthetic_designs_v1.json')
    parser.add_argument('--tiers', type=int, nargs='+', choices=[1, 2], default=[1])
    parser.add_argument('--descriptions', action='store_true')
    parser.add_argument('--view', choices=semantic.VIEWS, default='structured_no_name')
    parser.add_argument('--label', choices=['vanilla', 'mlm', 'semantic_mlm', 'tier1', 'tier12', 'tier12_description'], default='mlm')
    parser.add_argument('--revision', default='main', help='resolved downloaded files are hashed; pin a model revision for new downloads')
    parser.add_argument('--allow-expensive', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    if args.stage == 'download-model' and not args.allow_download:
        parser.error('download-model requires --allow-download')
    if args.stage in ('mlm', 'contrastive', 'export', 'evaluate') and not args.allow_expensive:
        parser.error(f'{args.stage} requires --allow-expensive; all model reads remain local-only')
    defaults = {'prepare': 'oracle', 'prepare-designs': 'synthetic_cards_v2', 'download-model': 'model_download_v2',
                'mlm': 'mlm_synthetic_cards_v1' if args.synthetic_cards else 'mlm_semantic' if args.semantic_records else 'mlm',
                'contrastive': 'contrastive_' + ('tier12_description' if args.descriptions else 'tier12' if 2 in args.tiers else 'tier1'),
                'export': 'semantic_' + args.label, 'evaluate': 'evaluation_' + args.label}
    output = args.output or v4.DEFAULT_ROOT / defaults[args.stage]
    if args.stage == 'prepare':
        semantic.prepare_oracle(args.oracle, output, args.deck_vocab, args.relations, args.benchmark, args.replacements, args.seed)
    elif args.stage == 'prepare-designs':
        from mtgdeck.static_synthetic_cards import prepare
        prepare(args.prepared, output, args.design_policy, args.seed)
    elif args.stage == 'download-model':
        semantic.download_model(output, args.revision)
    elif args.stage == 'mlm':
        semantic.train_mlm(args.prepared, args.model, output, args.seed, args.epochs, args.semantic_records,
                           args.max_length, args.batch_size, args.synthetic_cards, args.synthetic_fraction)
    elif args.stage == 'contrastive':
        semantic.train_contrastive(args.prepared, args.model, output, args.seed, args.epochs, tuple(args.tiers), args.descriptions, args.batch_size, args.max_length)
    elif args.stage == 'export':
        semantic.export_vectors(args.prepared, args.model, output, args.label, args.deck_vocab, args.view, args.batch_size, args.max_length)
    else:
        semantic.evaluate_semantic(args.prepared, args.vectors, output)
    print(output)


if __name__ == '__main__':
    main()
