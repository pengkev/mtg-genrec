"""Local operational tools. No scraping, cloud SDKs, or model dependencies."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from scrape import ROOT

FIXED = (
    'data/cooccurence/embedding_corpus.jsonl',
    'data/diverse_scraper.checkpoint.json',
    'data/format_corpora/.diverse_scraper.seen.sqlite3',
    'data/oracle_cards.jsonl.gz',
    'data/oracle_cards.jsonl.gz.metadata.json',
)
RETRY_DELAYS = (60, 300, 900)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ops(root):
    return Path(root) / 'logs/scraper'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def identity(pid):
    """PID plus kernel creation time prevents stale metadata targeting a reused PID."""
    if not pid:
        return None
    if os.name == 'nt':
        from ctypes import wintypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        handle = k.OpenProcess(0x1000 | 0x100000, False, pid)
        if not handle:
            return None
        try:
            if k.WaitForSingleObject(handle, 0) != 258:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return f'{pid}:{times[0].dwHighDateTime}:{times[0].dwLowDateTime}'
        finally:
            k.CloseHandle(handle)
    try:
        # comm may contain spaces/parentheses; starttime is field 22.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return f'{pid}:{fields[19]}'
    except (OSError, IndexError):
        return None


class Busy(RuntimeError):
    pass


class FileLock:
    """Kernel-owned byte lock; persistent inode must NEVER be unlinked as stale."""
    def __init__(self, path, create=True):
        self.path = Path(path)
        self.create = create
        self.file = None

    def __enter__(self):
        if self.create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b' if self.create else 'r+b')
        try:
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise Busy(f'Another operation owns {self.path}') from exc
        return self

    def __exit__(self, *exc):
        if self.file:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()
            self.file = None


def held(path):
    if not Path(path).exists():
        return False
    try:
        with FileLock(path, create=False):
            return False
    except Busy:
        return True


def ensure_integrity(root):
    if (ops(root) / 'restore-pending.json').exists():
        raise RuntimeError('Interrupted restore: run restore_scraper_state.ps1 -RecoverInterrupted before scraping/backing up.')


def recovery_paths(root):
    root = Path(root)
    corpora = sorted(p for p in (root / 'data/format_corpora').glob('*.jsonl')
                     if not p.name.lower().startswith('limited-'))
    if not corpora:
        raise ValueError('No constructed corpora found; refusing incomplete recovery set')
    result = [root / name for name in FIXED] + corpora
    for p in result:
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Missing/unsafe recovery member: {p}')
    # Closed SQLite should have no recovery journal left. Never ignore a WAL.
    db = root / FIXED[2]
    for suffix in ('-wal', '-shm', '-journal'):
        if Path(str(db) + suffix).exists():
            raise ValueError(f'SQLite transient file present: {db}{suffix}; inspect crash recovery first')
    return sorted(result)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root):
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def backup(root, destination):
    """Caller holds both gate and writer locks."""
    root, destination = Path(root), Path(destination).resolve()
    if destination.is_relative_to((root / 'data').resolve()):
        raise ValueError('Backup destination must be outside data/')
    paths = recovery_paths(root)
    target = destination / (dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, timestamp=now(), git_commit=git_commit(root), source_path=str(root.resolve()),
                    destination_path=str(target), files=[])
    for source in paths:
        rel = source.relative_to(root).as_posix()
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copy2(source, dst)
        sha = digest(source)
        if dst.stat().st_size != before.st_size or digest(dst) != sha or (source.stat().st_size, source.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f'Copy verification/source stability failed: {source}')
        manifest['files'].append(dict(path=rel, size_bytes=before.st_size, sha256=sha))
    if [p.relative_to(root).as_posix() for p in recovery_paths(root)] != [r['path'] for r in manifest['files']]:
        raise ValueError('Recovery membership changed during backup')
    atomic_json(target / 'manifest.json', manifest)
    validate(target, require_complete=False)
    atomic_json(target / 'COMPLETE.json', dict(timestamp=now(), manifest_sha256=digest(target / 'manifest.json'), files=len(paths)))
    atomic_json(ops(root) / 'last-backup.json', dict(timestamp=now(), path=str(target)))
    return target


def allowed_member(name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        return False
    p = Path(name)
    return name in FIXED or (p.as_posix() == name and len(p.parts) == 3 and p.parts[:2] == ('data', 'format_corpora') and
                            p.suffix == '.jsonl' and not p.name.lower().startswith('limited-'))


def validate(target, require_complete=True):
    target = Path(target).resolve()
    marker = read_json(target / 'COMPLETE.json') if require_complete else None
    if marker and digest(target / 'manifest.json') != marker['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    manifest = read_json(target / 'manifest.json')
    records = manifest['files']
    names = [r['path'] for r in records]
    if manifest.get('version') != 1 or len(names) != len({n.casefold() for n in names}) or (marker is not None and len(names) != marker['files']):
        raise ValueError('Invalid manifest version/count/duplicate paths')
    if not set(FIXED).issubset(names) or not any(n not in FIXED for n in names):
        raise ValueError('Incomplete recovery membership')
    errors = []
    for r in records:
        p = target / r['path']
        if not allowed_member(r['path']) or not p.resolve().is_relative_to(target) or p.is_symlink():
            errors.append(f'Unsafe member {r["path"]}')
        elif not p.is_file() or p.stat().st_size != r['size_bytes'] or digest(p) != r['sha256']:
            errors.append(f'Missing/corrupt member {r["path"]}')
    actual = {p.relative_to(target).as_posix() for p in (target / 'data').rglob('*') if p.is_file()}
    if actual != set(names):
        errors.append('Manifest does not match all backup data files')
    if errors:
        raise ValueError('\n'.join(errors))
    return manifest


def apply_restore(root, target):
    """Stage and verify everything before touching current files; journal blocks readers on crash."""
    root, target = Path(root), Path(target)
    manifest = validate(target)
    stage = ops(root) / ('restore-stage-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    for r in manifest['files']:
        dst = stage / r['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / r['path'], dst)
        if dst.stat().st_size != r['size_bytes'] or digest(dst) != r['sha256']:
            raise ValueError('Restore staging verification failed')
    wanted = {r['path'] for r in manifest['files']}
    # Remove later-generation extra corpora from live set by MOVING to retained rollback area.
    for current in (root / 'data/format_corpora').glob('*.jsonl'):
        if not current.name.lower().startswith('limited-') and current.relative_to(root).as_posix() not in wanted:
            dst = stage / 'displaced' / current.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(current, dst)
    for r in manifest['files']:
        dest = root / r['path']
        if not dest.resolve().is_relative_to(root.resolve()) or dest.is_symlink():
            raise ValueError('Unsafe restore destination')
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / r['path'], dest)
    for r in manifest['files']:
        if digest(root / r['path']) != r['sha256'] or (root / r['path']).stat().st_size != r['size_bytes']:
            raise ValueError('Restored member verification failed')


def restore(root, target=None, recover=False):
    journal = ops(root) / 'restore-pending.json'
    if recover:
        prior = read_json(journal)['pre_restore_backup']
        apply_restore(root, prior)
        journal.unlink()
        return prior
    ensure_integrity(root)
    validate(target)
    prior = backup(root, Path(root) / 'backups/scraper/pre-restore')
    atomic_json(journal, dict(timestamp=now(), pre_restore_backup=str(prior), requested_backup=str(Path(target).resolve())))
    try:
        apply_restore(root, target)
    except BaseException:
        # Keep journal even after rollback, so operator explicitly acknowledges interrupted restore.
        try:
            apply_restore(root, prior)
        except BaseException:
            pass
        raise
    journal.unlink()
    return prior


def free_bytes(root):
    data = Path(root) / 'data'
    return shutil.disk_usage(data if data.exists() else root).free


def disk_safe(free, minimum_gb):
    return free >= minimum_gb * 1024**3


def retry_delay(exit_code, attempt, manual=False, low_disk=False, integrity=False, maximum=3):
    if exit_code != 1 or manual or low_disk or integrity or attempt >= min(maximum, len(RETRY_DELAYS)):
        return None
    return RETRY_DELAYS[attempt]


def owner(root):
    p = ops(root) / 'owner.json'
    return read_json(p) if p.exists() else {}


def stop_requested(root, token):
    p = ops(root) / ('stop-' + token + '.json')
    return p.exists()


def request_stop(root, info):
    if not info.get('token'):
        raise ValueError('No identifiable scraper run')
    atomic_json(ops(root) / ('stop-' + info['token'] + '.json'), dict(timestamp=now(), reason='manual'))


def force_process(pid, expected):
    if not expected or identity(pid) != expected:
        raise RuntimeError('Process identity changed; refusing termination')
    if os.name == 'nt':
        from ctypes import wintypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        k.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k.OpenProcess(0x1001, False, pid)
        if not handle:
            raise RuntimeError('Cannot open owned process')
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)) or f'{pid}:{times[0].dwHighDateTime}:{times[0].dwLowDateTime}' != expected:
                raise RuntimeError('Process identity changed')
            if not k.TerminateProcess(handle, 1):
                raise RuntimeError('Termination failed')
        finally:
            k.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGKILL)


def stop(root, timeout=120, force=False):
    info = owner(root)
    if not (held(ops(root) / 'gate.lock') or held(ops(root) / 'writer.lock')):
        print('Scraper stopped; stale metadata is not a running owner.')
        return 0
    owner_valid = bool(info.get('identity')) and identity(info.get('pid')) == info.get('identity')
    child_valid = bool(info.get('scraper_identity')) and identity(info.get('scraper_pid')) == info.get('scraper_identity')
    if not info.get('token') or not (owner_valid or child_valid):
        raise RuntimeError('Lock active but owner identity cannot be verified; refusing stop')
    def state_stamps():
        return {name: ((Path(root) / name).stat().st_size, (Path(root) / name).stat().st_mtime_ns) if (Path(root) / name).exists() else None for name in FIXED[1:3]}
    before = state_stamps()
    request_stop(root, info)
    end = time.monotonic() + timeout
    while (held(ops(root) / 'gate.lock') or held(ops(root) / 'writer.lock')) and time.monotonic() < end:
        time.sleep(.25)
    if held(ops(root) / 'gate.lock') or held(ops(root) / 'writer.lock'):
        if not force:
            print('Graceful stop requested, still running; no process killed. Increase timeout or explicitly use -Force.')
            code = 2
        else:
            current = owner(root)
            if current.get('token') != info.get('token'):
                raise RuntimeError('Owner generation changed; refusing force')
            pid = current.get('scraper_pid') or current['pid']
            expected = current.get('scraper_identity') or current['identity']
            force_process(pid, expected)
            print('Forced termination of verified scraper; checkpoint may need recovery.')
            code = 1
    else:
        print('Scraper exited after graceful stop request.')
        code = 0
    after = state_stamps()
    for name in before:
        print(f'{name}: size/mtime changed during shutdown: {before[name] != after[name]}')
    return code


def status(root, health=False, stuck_hours=4):
    directory = ops(root)
    info = owner(root)
    active = held(directory / 'gate.lock') or held(directory / 'writer.lock')
    supervisor_alive = bool(info.get('identity')) and identity(info.get('pid')) == info.get('identity')
    scraper_alive = bool(info.get('scraper_identity')) and identity(info.get('scraper_pid')) == info.get('scraper_identity')
    valid = active and (supervisor_alive or scraper_alive)
    result = dict(running=active, supervisor_alive=supervisor_alive, scraper_alive=scraper_alive, lock='valid' if valid else 'active owner unknown' if active else 'stale/unowned' if info else 'unowned', owner=info, free_bytes=free_bytes(root))
    if info.get('started_at'):
        result['runtime_seconds'] = max(0, (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(info['started_at'])).total_seconds()) if active else None
    stats = {}
    for name in FIXED[:3]:
        p = Path(root) / name
        stats[name] = dict(size_bytes=p.stat().st_size, modified_at=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat()) if p.exists() else None
    corpora = [p for p in (Path(root) / 'data/format_corpora').glob('*.jsonl') if not p.name.lower().startswith('limited-')]
    result.update(files=stats, constructed_bytes=sum(p.stat().st_size for p in corpora), constructed_files=len(corpora))
    log = Path(info['log']) if info.get('log') else None
    lines = []
    if log and log.is_file():
        # Bounded tail even when a log is huge.
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 65536))
            lines = f.read().decode('utf-8', errors='replace').splitlines()
    result['latest_log_lines'] = lines[-20:]
    if health:
        for key in ('last-run', 'last-success', 'last-backup'):
            p = directory / (key + '.json')
            result[key] = read_json(p) if p.exists() else None
        result['latest_error_lines'] = [s for s in lines if any(t in s.lower() for t in ('error', 'exception', 'traceback', 'critical', 'failed'))][-20:]
        checkpoint = Path(root) / FIXED[1]
        result['checkpoint_age_seconds'] = time.time() - checkpoint.stat().st_mtime if checkpoint.exists() else None
        result['warnings'] = []
        if not result['last-backup'] or not (Path(result['last-backup']['path']) / 'COMPLETE.json').is_file():
            result['warnings'].append('No recorded successful backup exists (run validation to check recorded backups).')
        activity = [p.stat().st_mtime for p in corpora + [Path(root) / n for n in FIXED] if p.exists()]
        if log and log.exists():
            activity.append(log.stat().st_mtime)
        if active and activity and time.time() - max(activity) > stuck_hours * 3600:
            result['warnings'].append('No state/log activity within configured interval; scraper may be stuck.')
        if (directory / 'restore-pending.json').exists():
            result['warnings'].append('Interrupted restore blocks scraping; recover pre-restore generation.')
    print(json.dumps(result, indent=2))
    return result


def run(root, args):
    directory = ops(root)
    ensure_integrity(root)
    # An orphan child may outlive its wrapper. Never replace its ownership metadata.
    with FileLock(directory / 'writer.lock'):
        pass
    # Fail before launching: intact recovery membership, no SQLite recovery remnants.
    recovery_paths(root)
    if not disk_safe(free_bytes(root), args.minimum_free_gb):
        raise ValueError('Insufficient free disk space before start')
    log = directory / (dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + uuid.uuid4().hex[:8] + '.log')
    token = uuid.uuid4().hex
    info = dict(token=token, pid=os.getpid(), identity=identity(os.getpid()), wrapper_pid=args.wrapper_pid,
                started_at=now(), hostname=socket.gethostname(), log=str(log), arguments=args.scraper_args, git_commit=git_commit(root))
    atomic_json(directory / 'owner.json', info)
    deadline = time.monotonic() + args.max_runtime_hours * 3600 if args.max_runtime_hours else float('inf')
    manual = False
    def on_signal(*_):
        nonlocal manual
        manual = True
    old = signal.signal(signal.SIGINT, on_signal)
    child = None
    final_code = 2
    reason = 'startup failure'
    with log.open('a', encoding='utf-8', buffering=1) as stream:
        def emit(message):
            line = f'{now()} {message}'
            print(line, flush=True)
            stream.write(line + '\n')
        emit('START ' + json.dumps(info))
        try:
            attempt = 0
            while True:
                if manual or stop_requested(root, token) or time.monotonic() >= deadline:
                    final_code, reason = 0, 'manual stop or maximum runtime reached before restart'
                    break
                ensure_integrity(root)
                if not disk_safe(free_bytes(root), args.critical_free_gb):
                    final_code, reason = 2, 'critical disk space before restart'
                    break
                env = os.environ.copy()
                env.update(MTG_SCRAPER_TOKEN=token, PYTHONUNBUFFERED='1')
                flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                child = subprocess.Popen([sys.executable, '-u', '-m', 'scrape.scraper', *args.scraper_args], cwd=root, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', creationflags=flags)
                info.update(scraper_pid=child.pid, scraper_identity=identity(child.pid))
                atomic_json(directory / 'owner.json', info)
                emit(f'CHILD started pid={child.pid} identity={info["scraper_identity"]}')
                output = queue.Queue(maxsize=2048)
                def reader(proc):
                    for line in proc.stdout:
                        output.put(line.rstrip('\n'))
                    proc.stdout.close()
                    output.put(None)
                thread = threading.Thread(target=reader, args=(child,), daemon=True)
                thread.start()
                stopping = None
                forced = False
                reason = 'child exit'
                next_disk_check = 0
                eof = False
                while child.poll() is None or not eof:
                    try:
                        line = output.get(timeout=.2)
                        if line is None:
                            eof = True
                        else:
                            emit(line)
                    except queue.Empty:
                        pass
                    t = time.monotonic()
                    low = False
                    if t >= next_disk_check:
                        low = not disk_safe(free_bytes(root), args.critical_free_gb)
                        next_disk_check = t + args.disk_check_seconds
                    requested = manual or stop_requested(root, token)
                    if stopping is None and (requested or low or t >= deadline):
                        reason = 'manual stop' if requested else 'CRITICAL disk space' if low else 'maximum runtime'
                        emit('SHUTDOWN requested: ' + reason)
                        request_stop(root, info)
                        stopping = t
                    if stopping is not None and child.poll() is None and t - stopping >= args.shutdown_timeout:
                        emit('FORCED shutdown after graceful timeout')
                        child.kill()  # Popen handle targets this child, never arbitrary Python processes.
                        forced = True
                    if eof and child.poll() is not None:
                        break
                code = child.wait()
                child = None
                emit(f'CHILD exit_code={code}; shutdown={"forced" if forced else "graceful" if stopping is not None and code == 0 else "exited"}')
                if stopping is not None:
                    final_code = 1 if forced or code else 0
                    break
                wait = retry_delay(code, attempt, maximum=args.max_retries)
                if wait is None:
                    final_code = code
                    break
                attempt += 1
                emit(f'RETRY {attempt}/{args.max_retries} after {wait}s')
                until = min(time.monotonic() + wait, deadline)
                while time.monotonic() < until and not manual and not stop_requested(root, token):
                    time.sleep(.25)
        finally:
            # Never release gate while a child is still writing, including wrapper exceptions/Ctrl+C.
            if child is not None and child.poll() is None:
                request_stop(root, info)
                try:
                    child.wait(timeout=args.shutdown_timeout)
                    emit('Emergency cleanup: child exited after graceful request')
                except subprocess.TimeoutExpired:
                    emit('Emergency cleanup: FORCED shutdown')
                    child.kill()
                    child.wait()
            result = dict(started_at=info['started_at'], ended_at=now(), exit_code=final_code, reason=reason, log=str(log), arguments=args.scraper_args)
            emit('END ' + json.dumps(result))
            atomic_json(directory / 'last-run.json', result)
            if final_code == 0:
                atomic_json(directory / 'last-success.json', result)
            signal.signal(signal.SIGINT, old)
    return final_code


# Installed only for the CLI entry point, leaving imports/tests free of operational state.
_stop_check = lambda: False


def check_stop():
    if _stop_check():
        raise KeyboardInterrupt('Cooperative local stop requested')


def scraper_entry(main):
    try:
        return _scraper_entry(main)
    except (Busy, OSError, ValueError) as exc:
        print(f'Local ownership/integrity failure (no automatic retry): {exc}', file=sys.stderr)
        return 2


def _scraper_entry(main):
    global _stop_check
    root = ROOT
    token = os.environ.get('MTG_SCRAPER_TOKEN')
    with contextlib.ExitStack() as stack:
        if token:
            info = owner(root)
            if token != info.get('token') or identity(info.get('pid')) != info.get('identity') or not held(ops(root) / 'gate.lock'):
                raise Busy('Wrapper ownership could not be verified')
        else:
            stack.enter_context(FileLock(ops(root) / 'gate.lock'))
            token = uuid.uuid4().hex
            info = dict(token=token, pid=os.getpid(), identity=identity(os.getpid()), scraper_pid=os.getpid(),
                        scraper_identity=identity(os.getpid()), started_at=now(), hostname=socket.gethostname(), arguments=sys.argv[1:])
        stack.enter_context(FileLock(ops(root) / 'writer.lock'))
        if not os.environ.get('MTG_SCRAPER_TOKEN'):
            atomic_json(ops(root) / 'owner.json', info)
        ensure_integrity(root)
        stopped = False
        def signal_stop(*_):
            nonlocal stopped
            stopped = True
        old = signal.signal(signal.SIGINT, signal_stop)
        _stop_check = lambda: stopped or stop_requested(root, token)
        try:
            return main()
        except KeyboardInterrupt:
            print('Graceful shutdown: scraper cleanup completed.', flush=True)
            return 0
        except (ValueError, OSError) as exc:
            print(f'Integrity/local I/O failure (no automatic retry): {exc}', file=sys.stderr)
            return 2
        finally:
            _stop_check = lambda: False
            signal.signal(signal.SIGINT, old)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--max-runtime-hours', type=float, default=0)
    p.add_argument('--minimum-free-gb', type=float, default=10)
    p.add_argument('--critical-free-gb', type=float, default=5)
    p.add_argument('--disk-check-seconds', type=float, default=30)
    p.add_argument('--shutdown-timeout', type=float, default=120)
    p.add_argument('--max-retries', type=int, choices=range(4), default=3)
    p.add_argument('--wrapper-pid', type=int)
    p.add_argument('scraper_args', nargs=argparse.REMAINDER)
    for cmd in ('status', 'health'):
        p = sub.add_parser(cmd)
        p.add_argument('--stuck-hours', type=float, default=4)
    p = sub.add_parser('stop'); p.add_argument('--timeout', type=float, default=120); p.add_argument('--force', action='store_true')
    p = sub.add_parser('backup'); p.add_argument('--destination', default=str(ROOT / 'backups/scraper'))
    p = sub.add_parser('validate'); p.add_argument('backup')
    p = sub.add_parser('restore'); p.add_argument('backup', nargs='?'); p.add_argument('--recover', action='store_true')
    p = sub.add_parser('cleanup'); p.add_argument('--retain-days', type=int, default=30); p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.command in ('status', 'health'):
        status(ROOT, args.command == 'health', args.stuck_hours); return 0
    if args.command == 'stop':
        if args.timeout <= 0: raise ValueError('Timeout must be positive')
        return stop(ROOT, args.timeout, args.force)
    if args.command == 'validate':
        validate(args.backup); print('Backup valid: all hashes, sizes and recovery members verified.'); return 0
    if args.command == 'cleanup':
        if args.retain_days < 1: raise ValueError('Retention must be at least one day')
        current = owner(ROOT).get('log') if held(ops(ROOT) / 'gate.lock') else None
        for p in ops(ROOT).glob('*.log'):
            if p.is_symlink() or not p.resolve().is_relative_to(ops(ROOT).resolve()) or str(p) == current:
                continue
            if p.stat().st_mtime < time.time() - args.retain_days * 86400:
                print(('Would delete ' if args.dry_run else 'Delete ') + str(p))
                if not args.dry_run: p.unlink()
        return 0
    with FileLock(ops(ROOT) / 'gate.lock'):
        if args.command == 'run':
            if args.scraper_args[:1] == ['--']: args.scraper_args.pop(0)
            if args.max_runtime_hours < 0 or not 0 < args.critical_free_gb <= args.minimum_free_gb or args.disk_check_seconds <= 0 or args.shutdown_timeout <= 0:
                raise ValueError('Invalid runtime/disk/timeout settings')
            return run(ROOT, args)
        with FileLock(ops(ROOT) / 'writer.lock'):
            if args.command == 'backup':
                ensure_integrity(ROOT); print(backup(ROOT, args.destination))
            elif args.command == 'restore':
                if not args.recover and not args.backup: raise ValueError('Supply a backup directory or --recover')
                print('Pre-restore recovery generation: ' + str(restore(ROOT, args.backup, args.recover)))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(cli())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Operation refused/failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
