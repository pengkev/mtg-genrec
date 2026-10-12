"""Small shared contracts for isolated, write-once v4 research stages."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

from .artifacts import sha256_file
from .static_export import require_disjoint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / 'artifacts/card2vec/static_v4_multiview_v1'
DEFAULT_EXPORT = ROOT / 'artifacts/card2vec/static_export_v2'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def partition(family, seed=42):
    number = int(digest([seed, family])[:12], 16) / 16**12
    return 'train' if number < .8 else 'validation' if number < .9 else 'test'


def runtime():
    versions = {}
    for package in ('numpy', 'torch', 'transformers', 'accelerate', 'scipy', 'scikit-learn', 'nbformat'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    return {'python': platform.python_version(), 'packages': versions}


def code_hashes():
    sources = list(Path(__file__).parent.glob('static_*.py'))
    sources += [Path(__file__).with_name(n) for n in ('data.py', 'metadata.py', 'artifacts.py')]
    sources += list((ROOT / 'scripts').glob('*static*v4*.py'))
    sources += [ROOT / 'scripts' / f'run_static_{n}.py' for n in ('semantic', 'spellbook', 'multiview', 'v4_pipeline')]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sources if p.exists()}


def historical_roots():
    return [p for p in (ROOT / 'artifacts/card2vec').glob('static_*')
            if p.name.startswith(('static_v1', 'static_v2', 'static_v3', 'static_export_v1'))]


@contextmanager
def stage(output, config, inputs=()):
    output = require_disjoint(output, historical_roots())
    if output.exists():
        raise FileExistsError(f'Stage exists: {output}; choose a fresh version, including for incomplete stages')
    sources = {}
    for source in inputs:
        source = Path(source).resolve()
        require_disjoint(output, [source])
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        sources.update({str(p): sha256_file(p) for p in paths if p.is_file()})
    output.mkdir(parents=True)
    provenance = {'version': 'static_v4_multiview_v1', 'config': config,
                  'inputs': sources, 'implementation': code_hashes(), 'runtime': runtime()}
    write_json(output / 'provenance.json', provenance)
    yield output
    files = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'complete.json', {'files': files})


def validate_stage(path):
    """Offline integrity check; historical code hashes describe, never rewrite, a stage."""
    path = Path(path)
    for relative, expected in read_json(path / 'complete.json')['files'].items():
        file = (path / relative).resolve()
        if path.resolve() not in file.parents or sha256_file(file) != expected:
            raise ValueError(f'Stage hash/path mismatch: {relative}')
    return read_json(path / 'provenance.json')


class Families:
    """Connected components for overlapping supervision/family identifiers."""
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            item, self.parent[item] = self.parent[item], root
        return root

    def join(self, items):
        roots = sorted({self.find(item) for item in items})
        if roots:
            for root in roots[1:]:
                self.parent[root] = roots[0]
