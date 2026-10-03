"""Offline operational tests; every mutation uses pytest temporary directories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scrape import operations as op


@pytest.fixture
def recovery(tmp_path):
    for name in (*op.FIXED, 'data/format_corpora/modern.jsonl', 'data/format_corpora/deckbox.jsonl'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'{"test":1}\n')
    (tmp_path / 'data/format_corpora/limited-sealed.jsonl').write_bytes(b'limited untouched')
    return tmp_path


def test_lock_refuses_live_and_reclaims_stale_metadata(tmp_path):
    lock = tmp_path / 'owner.lock'
    with op.FileLock(lock):
        assert op.held(lock)
        with pytest.raises(op.Busy):
            with op.FileLock(lock):
                pass
    assert lock.exists()  # Never unlink an inode another contender could hold.
    assert not op.held(lock)
    with op.FileLock(lock):
        assert op.held(lock)


def test_kernel_releases_lock_after_process_death(tmp_path):
    lock = tmp_path / 'crashed.lock'
    code = 'import time; from scrape.operations import FileLock; from pathlib import Path; lock=FileLock(Path(__import__("sys").argv[1])); lock.__enter__(); print("ready", flush=True); time.sleep(60)'
    child = subprocess.Popen([sys.executable, '-c', code, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        assert op.held(lock)
    finally:
        child.kill(); child.wait(); child.stdout.close()
    deadline = time.monotonic() + 3
    while op.held(lock) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not op.held(lock)
    assert op.identity(child.pid) is None


def test_force_refuses_reused_pid():
    with pytest.raises(RuntimeError, match='identity'):
        op.force_process(os.getpid(), 'different-creation-time')


def test_recovery_members_include_hidden_state_exclude_limited(recovery):
    names = {p.relative_to(recovery).as_posix() for p in op.recovery_paths(recovery)}
    assert set(op.FIXED) <= names
    assert len(names) == 7
    assert not any('limited-' in n for n in names)


def test_missing_member_and_sqlite_wal_refused(recovery):
    wal = Path(str(recovery / op.FIXED[2]) + '-wal')
    wal.write_bytes(b'uncommitted')
    with pytest.raises(ValueError, match='transient'):
        op.recovery_paths(recovery)
    wal.unlink()
    (recovery / op.FIXED[1]).unlink()
    with pytest.raises(ValueError, match='Missing'):
        op.recovery_paths(recovery)


def test_backup_manifest_and_hashes(recovery, tmp_path):
    target = op.backup(recovery, tmp_path / 'backups')
    manifest = op.validate(target)
    assert len(manifest['files']) == 7
    assert {r['path'] for r in manifest['files']} == {p.relative_to(recovery).as_posix() for p in op.recovery_paths(recovery)}
    for r in manifest['files']:
        assert r['size_bytes'] == (recovery / r['path']).stat().st_size
        assert r['sha256'] == op.digest(recovery / r['path'])
    assert op.read_json(target / 'COMPLETE.json')['manifest_sha256'] == op.digest(target / 'manifest.json')


def test_partial_backup_never_gets_complete(recovery, monkeypatch):
    original = op.shutil.copy2
    copies = 0
    def fail(src, dest):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError('disk full')
        return original(src, dest)
    monkeypatch.setattr(op.shutil, 'copy2', fail)
    dest = recovery / 'backups'
    with pytest.raises(OSError):
        op.backup(recovery, dest)
    target, = dest.iterdir()
    assert not (target / 'COMPLETE.json').exists()
    with pytest.raises(FileNotFoundError):
        op.validate(target)


def test_corrupt_hash_and_missing_member_rejected(recovery):
    target = op.backup(recovery, recovery / 'backups')
    file = target / op.FIXED[0]
    file.write_bytes(b'{"test":2}\n')  # Same length, different hash.
    with pytest.raises(ValueError, match='corrupt'):
        op.validate(target)
    file.unlink()
    with pytest.raises(ValueError, match='Missing'):
        op.validate(target)


def test_manifest_tampering_rejected(recovery):
    target = op.backup(recovery, recovery / 'backups')
    m = op.read_json(target / 'manifest.json')
    m['files'].pop()
    op.atomic_json(target / 'manifest.json', m)
    with pytest.raises(ValueError, match='Manifest hash'):
        op.validate(target)


@pytest.mark.parametrize('name', ['../data/format_corpora/a.jsonl', 'data/format_corpora/limited-sealed.jsonl', 'C:/evil', '/tmp/evil', 'data/format_corpora/../evil.jsonl', 'data\\format_corpora\\x.jsonl'])
def test_restore_member_allowlist(name):
    assert not op.allowed_member(name)


def test_complete_marker_cannot_validate_missing_fixed_set(recovery):
    target = op.backup(recovery, recovery / 'backups')
    m = op.read_json(target / 'manifest.json')
    m['files'] = [r for r in m['files'] if r['path'] != op.FIXED[0]]
    op.atomic_json(target / 'manifest.json', m)
    op.atomic_json(target / 'COMPLETE.json', dict(manifest_sha256=op.digest(target / 'manifest.json'), files=len(m['files'])))
    with pytest.raises(ValueError, match='Incomplete'):
        op.validate(target)


def test_complete_marker_written_only_after_validation(recovery, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('validation failed')
    monkeypatch.setattr(op, 'validate', fail)
    with pytest.raises(ValueError):
        op.backup(recovery, recovery / 'backups')
    assert not list((recovery / 'backups').rglob('COMPLETE.json'))


def test_backup_lock_refusal(recovery, monkeypatch):
    monkeypatch.setattr(op, 'ROOT', recovery)
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        with pytest.raises(op.Busy):
            op.cli(['backup'])
    assert not (recovery / 'backups').exists()


def test_restore_entire_generation_and_prebackup(recovery):
    target = op.backup(recovery, recovery / 'backups')
    checkpoint = recovery / op.FIXED[1]
    checkpoint.write_bytes(b'new state')
    extra = recovery / 'data/format_corpora/legacy.jsonl'
    extra.write_bytes(b'later corpus')
    pre = op.restore(recovery, target)
    assert (pre / op.FIXED[1]).read_bytes() == b'new state'
    assert (pre / 'data/format_corpora/legacy.jsonl').exists()
    assert checkpoint.read_bytes() == b'{"test":1}\n'
    assert not extra.exists()
    assert (recovery / 'data/format_corpora/limited-sealed.jsonl').read_bytes() == b'limited untouched'
    assert not (op.ops(recovery) / 'restore-pending.json').exists()


def test_failed_restore_blocks_run_until_explicit_rollback(recovery, monkeypatch):
    target = op.backup(recovery, recovery / 'backups')
    (recovery / op.FIXED[1]).write_bytes(b'latest')
    original = op.apply_restore
    def fail(*_):
        raise OSError('simulated interrupted replacement')
    monkeypatch.setattr(op, 'apply_restore', fail)
    with pytest.raises(OSError):
        op.restore(recovery, target)
    with pytest.raises(RuntimeError, match='Interrupted restore'):
        op.ensure_integrity(recovery)
    monkeypatch.setattr(op, 'apply_restore', original)
    op.restore(recovery, recover=True)
    assert (recovery / op.FIXED[1]).read_bytes() == b'latest'
    op.ensure_integrity(recovery)


def test_disk_threshold_and_bounded_retry():
    assert op.disk_safe(10 * 1024**3, 10)
    assert not op.disk_safe(10 * 1024**3 - 1, 10)
    assert [op.retry_delay(1, i) for i in range(4)] == [60, 300, 900, None]
    for code in (0, 2, 130):
        assert op.retry_delay(code, 0) is None
    for reason in ('manual', 'low_disk', 'integrity'):
        assert op.retry_delay(1, 0, **{reason: True}) is None
    assert op.retry_delay(1, 0, maximum=0) is None


def test_stop_check_uses_existing_keyboard_interrupt_cleanup(monkeypatch):
    monkeypatch.setattr(op, '_stop_check', lambda: True)
    with pytest.raises(KeyboardInterrupt):
        op.check_stop()


def test_low_disk_refuses_launch(recovery, monkeypatch):
    monkeypatch.setattr(op, 'free_bytes', lambda root: 0)
    args = SimpleNamespace(minimum_free_gb=10)
    with pytest.raises(ValueError, match='disk'):
        op.run(recovery, args)
    assert not (op.ops(recovery) / 'owner.json').exists()


def fake_scraper(recovery, body):
    module = recovery / 'scrape'
    module.mkdir()
    (module / '__init__.py').write_text('')
    (module / 'scraper.py').write_text(body)
    return SimpleNamespace(max_runtime_hours=0.0001, minimum_free_gb=.000001, critical_free_gb=.000001,
                           disk_check_seconds=.01, shutdown_timeout=3, max_retries=0, wrapper_pid=os.getpid(), scraper_args=['--offline-test'])


def test_supervisor_runtime_graceful_stop_and_logging(recovery):
    args = fake_scraper(recovery, '''import os,time,pathlib
p=pathlib.Path('logs/scraper') / ('stop-'+os.environ['MTG_SCRAPER_TOKEN']+'.json')
print('offline child ready',flush=True)
while not p.exists(): time.sleep(.02)
print('offline child flushed',flush=True)
''')
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        assert op.run(recovery, args) == 0
    last = op.read_json(op.ops(recovery) / 'last-run.json')
    log = Path(last['log']).read_text()
    assert 'offline child flushed' in log
    assert 'shutdown=graceful' in log
    assert 'START' in log and 'END' in log
    assert last['reason'] == 'maximum runtime'


def test_supervisor_forces_only_its_unresponsive_child(recovery):
    args = fake_scraper(recovery, 'import time; print("offline blocked",flush=True); time.sleep(60)')
    args.shutdown_timeout = .1
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        assert op.run(recovery, args) == 1
    last = op.read_json(op.ops(recovery) / 'last-run.json')
    assert 'FORCED' in Path(last['log']).read_text()
    assert not (op.ops(recovery) / 'last-success.json').exists()


def test_orphan_writer_prevents_wrapper_metadata_replacement(recovery):
    op.atomic_json(op.ops(recovery) / 'owner.json', {'token':'orphan'})
    with op.FileLock(op.ops(recovery) / 'writer.lock'):
        with pytest.raises(op.Busy):
            op.run(recovery, SimpleNamespace())
    assert op.owner(recovery)['token'] == 'orphan'


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell parser/forwarding validation')
def test_powershell_syntax_and_argument_forwarding(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    ps = shutil.which('powershell.exe')
    command = "$bad = @(); Get-ChildItem -LiteralPath '" + str(scripts).replace("'", "''") + "' -Filter '*.ps1' | ForEach-Object { $tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors); $bad += $errors }; if ($bad.Count) { $bad | Out-String | Write-Output; exit 1 }"
    subprocess.run([ps, '-NoProfile', '-Command', command], check=True, capture_output=True, text=True)
    shutil.copy2(scripts / 'run_scraper.ps1', tmp_path / 'run_scraper.ps1')
    (tmp_path / 'scraper_common.ps1').write_text('function Invoke-ScraperOperation { param([string[]]$OperationArgs) ConvertTo-Json -Compress -InputObject $OperationArgs; $script:OperationExitCode = 0 }')
    result = subprocess.run([ps, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(tmp_path / 'run_scraper.ps1'), '-MaxRuntimeHours', '12', '--sources', 'moxfield', 'mtgtop8', 'deckbox', '--max-pages-per-format', '100'], check=True, capture_output=True, text=True)
    args = json.loads(result.stdout.strip())
    assert args[3:] == ['--max-runtime-hours', '12', '--', '--sources', 'moxfield', 'mtgtop8', 'deckbox', '--max-pages-per-format', '100']


def test_direct_entry_refuses_existing_writer_without_replacing_owner(recovery, monkeypatch):
    monkeypatch.setattr(op, 'ROOT', recovery)
    monkeypatch.delenv('MTG_SCRAPER_TOKEN', raising=False)
    op.atomic_json(op.ops(recovery) / 'owner.json', {'token': 'existing'})
    with op.FileLock(op.ops(recovery) / 'writer.lock'):
        assert op.scraper_entry(lambda: pytest.fail('must not start')) == 2
    assert op.owner(recovery)['token'] == 'existing'


def test_cooperative_entry_runs_finally_and_releases_locks(recovery, monkeypatch):
    monkeypatch.setattr(op, 'ROOT', recovery)
    monkeypatch.delenv('MTG_SCRAPER_TOKEN', raising=False)
    flushed = []
    def main():
        op.request_stop(recovery, op.owner(recovery))
        try:
            op.check_stop()
        finally:
            flushed.append(True)
    assert op.scraper_entry(main) == 0
    assert flushed == [True]
    assert not op.held(op.ops(recovery) / 'gate.lock')
    assert not op.held(op.ops(recovery) / 'writer.lock')


def test_stop_timeout_never_kills_without_force(recovery, monkeypatch):
    op.atomic_json(op.ops(recovery) / 'owner.json', dict(token='test-stop', pid=os.getpid(), identity=op.identity(os.getpid())))
    monkeypatch.setattr(op, 'force_process', lambda *_: pytest.fail('must not kill'))
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        assert op.stop(recovery, timeout=.01) == 2
    assert op.stop_requested(recovery, 'test-stop')


def test_status_and_health_are_read_only(recovery):
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in recovery.rglob('*') if p.is_file()}
    result = op.status(recovery, health=True)
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in recovery.rglob('*') if p.is_file()}
    assert before == after
    assert result['constructed_files'] == 2
    assert not result['running']
    assert result['warnings']


def test_retention_dryrun_and_dataset_safety(recovery, monkeypatch):
    monkeypatch.setattr(op, 'ROOT', recovery)
    folder = op.ops(recovery)
    folder.mkdir(parents=True)
    old = folder / 'old.log'
    active = folder / 'active.log'
    for p in (old, active):
        p.write_text('old')
        os.utime(p, (0, 0))
    op.atomic_json(folder / 'owner.json', {'log': str(active)})
    with op.FileLock(folder / 'gate.lock'):
        op.cli(['cleanup', '--dry-run'])
        assert old.exists()
        op.cli(['cleanup'])
        assert not old.exists()
        assert active.exists()
    assert len(op.recovery_paths(recovery)) == 7


def test_supervisor_retry_budget_is_bounded(recovery, monkeypatch):
    args = fake_scraper(recovery, '''from pathlib import Path
p=Path('attempts.txt')
p.write_text(p.read_text()+'x' if p.exists() else 'x')
raise SystemExit(1)
''')
    args.max_runtime_hours = 0
    args.max_retries = 3
    monkeypatch.setattr(op, 'RETRY_DELAYS', (.01, .01, .01))
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        assert op.run(recovery, args) == 1
    assert (recovery / 'attempts.txt').read_text() == 'xxxx'


def test_low_disk_during_run_requests_stop_without_retry(recovery, monkeypatch):
    args = fake_scraper(recovery, '''import os,time,pathlib
p=pathlib.Path('logs/scraper')/('stop-'+os.environ['MTG_SCRAPER_TOKEN']+'.json')
while not p.exists(): time.sleep(.01)
''')
    args.max_runtime_hours = 0
    values = iter([100 * 1024**3, 100 * 1024**3])
    monkeypatch.setattr(op, 'free_bytes', lambda root: next(values, 0))
    with op.FileLock(op.ops(recovery) / 'gate.lock'):
        assert op.run(recovery, args) == 0
    last = op.read_json(op.ops(recovery) / 'last-run.json')
    assert last['reason'] == 'CRITICAL disk space'
    assert 'RETRY' not in Path(last['log']).read_text()
