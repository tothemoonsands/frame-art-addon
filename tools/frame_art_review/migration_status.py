"""Read-only background progress observer; never starts or resumes a migration."""
import json
import subprocess
import threading
import time

from store import ROOT
from ha_connection import command as ssh_command

_snapshot = {}
_lock = threading.Lock()


def load(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def merge(local, observed):
    if not local:
        return None
    result = dict(local)
    result.update({k:v for k,v in observed.items() if k not in ('remote_status','control')})
    status, control = observed.get('remote_status'), observed.get('control') or {}
    if status:
        result.update(status)
        if control.get('paused') and status.get('phase') != 'complete':
            result.update(phase='paused', error=control.get('error') or status.get('error'))
    elif control.get('error'):
        result.update(phase='paused', error=control['error'])
    # probe_ok + paused is the intentional staging hold, not an execution error.
    return result


def status():
    local = load(ROOT/'reports/migration-live.json', None)
    with _lock:
        observed = dict(_snapshot)
    return merge(local, observed)


def observe():
    previous = None
    failures = 0
    while True:
        if (ROOT/'reports/migration-live.json').exists():
            try:
                raw = subprocess.check_output(ssh_command()+['python3 -'], input='''
import json,os
from pathlib import Path
control=Path('/share/frame_art_migration/active.json')
c=json.loads(control.read_text()) if control.exists() else {}
p=Path('/share/frame_art_migration/runs/original-finish-20260909/status.json')
s=json.loads(p.read_text()) if p.exists() else None
size=0
for directory,_,files in os.walk('/media/frame_art_release'):
 for filename in files:
  try:size+=(Path(directory)/filename).stat().st_size
  except FileNotFoundError:pass
print(json.dumps(dict(staged_bytes=size,remote_status=s,control=c)))
''', text=True, timeout=15, stderr=subprocess.PIPE)
                observed = json.loads(raw)
                failures = 0
                stamp = time.time()
                release = load(ROOT/'release-original-finish/release.json', {})
                observed['transfer_total_bytes'] = release.get('summary',{}).get('layer_bytes')
                observed['observed_at'] = stamp
                if previous:
                    elapsed = stamp - previous['observed_at']
                    delta = observed['staged_bytes'] - previous['staged_bytes']
                    if elapsed > 0 and delta >= 0:
                        observed['transfer_bytes_per_second'] = delta / elapsed
                previous = observed
                with _lock:
                    _snapshot.clear()
                    _snapshot.update(observed)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                failures += 1
                reason = ('SSH status check timed out' if isinstance(exc, subprocess.TimeoutExpired)
                          else 'SSH status check failed' if isinstance(exc, subprocess.CalledProcessError)
                          else 'Status reading failed')
                with _lock:
                    _snapshot['poll_failures'] = failures
                    _snapshot['last_poll_error'] = reason
                    if failures >= 2:
                        _snapshot['connection_error'] = reason + '; showing the last reading and retrying. This does not pause the transfer.'
                print(f'Migration monitor: {reason} (consecutive failures: {failures})', flush=True)
        time.sleep(30)


def start():
    threading.Thread(target=observe, daemon=True, name='migration-status').start()
