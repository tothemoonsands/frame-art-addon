"""Stage the frozen release, then start its already-authorized HA migration.

This transfers files to HA, never directly to the TV. A paused/error probe is not
cleared automatically. Review choices and the release hash are checked again.
"""
import fcntl
import hashlib
import json
import subprocess
import shlex
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import curation
from store import ROOT
from ha_connection import command as ssh_command

SSH=ssh_command()
RELEASE=ROOT/'release-original-finish'
REPORT=ROOT/'reports/migration-live.json'
REMOTE_RUN='/share/frame_art_migration/runs/original-finish-20260909'


def report(**values):
    current=json.loads(REPORT.read_text()) if REPORT.exists() else {}
    current.update(values,checked_at=time.time())
    tmp=REPORT.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2));tmp.replace(REPORT)


def remote(script):
    return subprocess.check_output(SSH+['python3 -'],input=script,text=True,timeout=45)


def main():
    with (ROOT/'release-staging.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=json.loads((RELEASE/'release.json').read_text())
        digest=hashlib.sha256((RELEASE/'release.json').read_bytes()).hexdigest()
        report(phase='staging',completed=0,total=len(frozen['entries']),error=None)
        def transfer(number):
            files=sorted({v['path'] for row in frozen['entries'][number::4] for v in row['files'].values()})
            if number==0:files+=['release.json','confirmed-reconciliation.json']
            listing=ROOT/'logs'/f'release-shard-{number}.txt'
            listing.write_text('\n'.join(files)+'\n')
            with (ROOT/'logs'/f'release-shard-{number}.log').open('a') as log:
                subprocess.run(['rsync','-a','--partial','--stats','-e',shlex.join(SSH[:-1]),
                    '--files-from='+str(listing),str(RELEASE)+'/', 'root@192.168.1.202:/media/frame_art_release/'],
                    stdout=log,stderr=log,check=True)
            print(f'Transfer shard {number+1}/4 complete',flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(transfer,range(4)))
        review=curation.reconciliation()
        if not review['review_complete'] or review['render_style']['shadow']!='original' or review['render_style']['feather']!=0:
            raise ValueError('Review/finish changed during staging; migration remains held')
        if {k:a['selection']['asset'] for k,a in review['entries'].items()}!=frozen['review_choices']:
            raise ValueError('Artwork selections changed during staging; migration remains held')
        if hashlib.sha256((RELEASE/'release.json').read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen release changed during staging')
        print(remote(f'''
import hashlib,json,subprocess
from pathlib import Path
p=Path('/share/frame_art_migration/active.json');c=json.loads(p.read_text())
if not c.get('probe_ok') or c.get('error'):raise SystemExit('TV connection probe must succeed before migration starts')
version=json.loads(subprocess.check_output(['ha','apps','info','ad1c2f89_frame_art_uploader_ai','--raw-json']))['data']['version']
if version!='4.1.1':raise SystemExit('Required add-on version is not installed')
if hashlib.sha256(Path('/media/frame_art_release/release.json').read_bytes()).hexdigest()!={digest!r}:raise SystemExit('Remote release changed')
journal=Path(c['run'])/'journal.json'
if journal.exists():raise SystemExit('Migration already has a journal; inspect its state rather than restarting the staging handoff')
c.update(paused=False,probe_only=False)
t=p.with_suffix('.tmp');t.write_text(json.dumps(c));t.replace(p)
print('Verified stage complete; authorized migration started')
'''),flush=True)
        report(phase='verifying',error=None)
        while True:
            try:
                result=json.loads(remote(f'''
import json
from pathlib import Path
p=Path({REMOTE_RUN!r})/'status.json'
c=Path('/share/frame_art_migration/active.json')
status=json.loads(p.read_text()) if p.exists() else {{'phase':'starting'}}
if c.exists():
 control=json.loads(c.read_text())
 if control.get('paused'):status.update(phase='paused',error=control.get('error'))
print(json.dumps(status))
'''))
                report(**result)
                print(json.dumps(result),flush=True)
                if result.get('phase') in ('complete','paused'):return
            except subprocess.SubprocessError as exc:
                report(connection_error=str(exc))
            time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        report(phase='staging_paused',error=str(exc))
        raise
