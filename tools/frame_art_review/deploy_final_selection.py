"""Resumable deployment of the frozen final-23 artwork selections."""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from samsungtvws import SamsungTVWS

from ha_connection import command

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frame_art_uploader_ai.cover_art import compress_png_path_to_jpeg_max_bytes

ROOT = Path('/Users/jsands/Documents/Frame Art Review')
COMPARE = ROOT / 'reports/final-23-comparison'
OUT = COMPARE / 'deployment'
REMOTE = '/share/frame_art_migration/runs/final-23-selection-20260910'
CONTROL = '/share/frame_art_migration/active.json'
APP = 'ad1c2f89_frame_art_uploader_ai'
TV = '192.168.1.38'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def remote(script, timeout=90):
    return subprocess.check_output(command() + ['python3 -'], input=script, text=True, timeout=timeout)


def load_inputs():
    manifest = json.loads((COMPARE / 'manifest.json').read_text())
    selections = json.loads((COMPARE / 'selections.json').read_text())
    if len(selections.get('choices', {})) != 23:
        raise RuntimeError('All 23 comparison choices must be saved before deployment')
    queue = json.loads((ROOT / 'reports/manual-fix-review-20260910/queue.json').read_text())['queue']
    by_id = {row['id']: row for row in queue}
    return manifest['albums'], selections, by_id


def prepare():
    albums, selections, by_id = load_inputs()
    OUT.mkdir(exist_ok=True)
    entries = []
    for album in albums:
        choice = selections['choices'][str(album['id'])]
        selected = Path(choice['path'])
        if sha(selected) != choice['sha256']:
            raise RuntimeError(f'Selected file changed: {selected}')
        key = by_id[album['id']]['cache_key']
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        with Image.open(selected) as image:
            if image.size != (3840, 2160):
                raise RuntimeError(f'Unexpected selected image size: {selected} {image.size}')
            image.convert('RGB').save(folder / 'widescreen.png', compress_level=1)
        if selected.suffix.lower() in {'.jpg', '.jpeg'} and selected.stat().st_size <= 4 * 1024**2:
            shutil.copy2(selected, folder / 'compressed.jpg')
        else:
            ok, size = compress_png_path_to_jpeg_max_bytes(folder / 'widescreen.png', folder / 'compressed.jpg')
            if not ok or size > 4 * 1024**2:
                raise RuntimeError(f'Could not compress selected image: {selected}')
        background = None
        source = ROOT / 'release-original-finish' / key / 'source.png'
        if not source.exists():
            source = ROOT / 'release-original-finish' / key / 'source.jpg'
        recipe = {
            'mode': 'final_manual_selection', 'album_id': album['id'], 'key': key,
            'artist': album['artist'], 'title': album['title'], 'choice': choice['choice'],
            'selected_path': str(selected), 'selected_sha256': choice['sha256'],
            'final_png_sha256': sha(folder / 'widescreen.png'),
            'final_jpeg_sha256': sha(folder / 'compressed.jpg'),
            'approval': 'User completed final 23 comparison and requested deployment',
            'deployed': False,
        }
        if choice.get('recipe'):
            repair_path = Path(choice['recipe'])
            repair = json.loads(repair_path.read_text())
            background = repair_path.parent / repair['background']
            source = repair_path.parent / repair['source']
            recipe.update(repair_recipe=str(repair_path), repair=repair)
        elif choice['choice'] == 'original':
            background = Path('/Users/jsands/Documents/Code/frame-music-local-art-generation/python/out/background') / f'{key}__3840x2160__background.png'
            recipe['mode'] = 'selected_pre_upgrade_original'
        else:
            release_recipe = ROOT / 'release-original-finish' / key / 'approved-recipe.json'
            if release_recipe.exists():
                recipe['release_recipe'] = json.loads(release_recipe.read_text())
            release_bg = ROOT / 'release-original-finish' / key / 'background.png'
            if release_bg.exists():
                background = release_bg
        if background and background.exists():
            shutil.copy2(background, folder / 'background.png')
            recipe['background_sha256'] = sha(folder / 'background.png')
        if source.exists():
            target = folder / ('source' + source.suffix.lower())
            shutil.copy2(source, target)
            recipe['source_file'] = target.name
            recipe['source_sha256'] = sha(target)
        atomic_json(folder / 'recipe.json', recipe)
        entries.append({
            'number': album['number'], 'album_id': album['id'], 'key': key,
            'artist': album['artist'], 'title': album['title'], 'choice': choice['choice'],
            'selected_sha256': choice['sha256'], 'jpeg_sha256': recipe['final_jpeg_sha256'],
            'folder': str(folder), 'status': 'prepared',
        })
    plan = {'format': 'frame-art-final-selection-deploy-v1', 'remote': REMOTE,
            'selection_revision': selections['revision'], 'entries': entries}
    atomic_json(OUT / 'plan.json', plan)
    return plan


def get_remote_catalog(keys):
    script = """import json,pathlib
c=json.loads(pathlib.Path('/share/frame_art_music_catalog.json').read_text())['entries']
print(json.dumps({k:c.get(k+'__3840x2160.jpg') for k in KEYS}))
""".replace('KEYS', repr(keys))
    return json.loads(remote(script))


def connect_art():
    return SamsungTVWS(TV, port=8001, timeout=15).art(timeout=15)


def inventory():
    last = None
    for _ in range(3):
        art = connect_art()
        try:
            result = art.available()
            if not isinstance(result, list):
                raise RuntimeError('TV returned invalid inventory')
            return {item['content_id'] for item in result}
        except Exception as exc:
            last = exc
        finally:
            art.close()
        time.sleep(3)
    raise last


def mutate(action, old_id=None, image=None, before=None):
    """One TV mutation per connection; resolve uncertain upload from inventory delta."""
    art = connect_art()
    try:
        if action == 'delete':
            return art.delete(old_id)
        return art.upload(Path(image).read_bytes(), file_type='JPEG', matte='none')
    except Exception:
        after = inventory()
        if action == 'delete' and old_id not in after:
            return None
        if action == 'upload':
            added = after - before
            if len(added) == 1:
                return added.pop()
        raise
    finally:
        art.close()


def initialize_remote(plan, catalog):
    replace = [entry for entry in plan['entries'] if catalog[entry['key']]['source_hash'] != entry['jpeg_sha256']]
    old_ids = {catalog[e['key']]['content_id'] for e in replace}
    ownership = {}
    full = json.loads(remote("import json,pathlib\nprint(pathlib.Path('/share/frame_art_music_catalog.json').read_text())"))['entries']
    for filename, value in full.items():
        cid = value.get('content_id')
        if cid in old_ids and value.get('state', 'active') == 'active':
            ownership.setdefault(cid, []).append(filename)
    shared = {cid: names for cid, names in ownership.items() if len(names) != 1}
    if shared:
        raise RuntimeError(f'Target TV IDs have unexpected ownership: {shared}')
    init = """import pathlib,json,shutil,hashlib
run=pathlib.Path(RUN);control=pathlib.Path(CONTROL)
if control.exists():
    active=json.loads(control.read_text())
    assert active.get('run')==RUN, active
else:
    run.mkdir(parents=True,exist_ok=True)
    control.write_text(json.dumps({'paused':True,'run':RUN,'release':RUN,'reason':'Final 23 user selections'}))
music=pathlib.Path('/media/frame_ai/music');share=pathlib.Path('/share')
paths=[]
for key in KEYS:
    paths += [music/'widescreen-compressed'/(key+'__3840x2160.jpg'),music/'widescreen'/(key+'__3840x2160.png'),music/'background'/(key+'__3840x2160__background.png'),music/'recipes'/(key+'.json')]
paths += [music/'index.json',music/'manifest.json']+[p for p in share.glob('frame_art*json') if p.is_file()]+[share/'frame_art_display/current.json']
checks={}
for p in dict.fromkeys(paths):
    if not p.exists():continue
    target=run/'backup'/p.relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(p,target)
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    checks[str(p)]=digest
(run/'checksums.json').write_text(json.dumps(checks,indent=2))
print('Backed up',len(checks),'files')
""".replace('RUN', repr(REMOTE)).replace('CONTROL', repr(CONTROL)).replace('KEYS', repr([e['key'] for e in replace]))
    print(remote(init), flush=True)
    subprocess.run(command() + ['ha apps stop ' + APP], check=True, timeout=60)
    subprocess.run(['rsync', '-a', '-e', shlex.join(command()[:-1]), str(OUT) + '/',
                    'root@192.168.1.202:' + REMOTE + '/approved/'], check=True)
    return replace, old_ids


def commit_remote(entry, old_id, new_id):
    script = """import json,pathlib,shutil,hashlib,os,datetime
run=pathlib.Path(RUN);src=run/'approved'/KEY;music=pathlib.Path('/media/frame_ai/music');share=pathlib.Path('/share')
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();recipe=json.loads((src/'recipe.json').read_text())
def atomic(p,value):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def replace(value):
    if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
    if isinstance(value,list):return [replace(v) for v in value]
    return NEW if value==OLD else value
targets=[('compressed.jpg',music/'widescreen-compressed'/(KEY+'__3840x2160.jpg')),('widescreen.png',music/'widescreen'/(KEY+'__3840x2160.png')),('recipe.json',music/'recipes'/(KEY+'.json'))]
if (src/'background.png').exists():targets.append(('background.png',music/'background'/(KEY+'__3840x2160__background.png')))
for name,target in targets:
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.selection-tmp');shutil.copy2(src/name,temp);temp.replace(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256((src/name).read_bytes()).hexdigest()
paths=[music/'index.json',music/'manifest.json']+[p for p in share.glob('frame_art*json') if p.is_file()]+[share/'frame_art_display/current.json']
for p in dict.fromkeys(paths):
    if not p.exists():continue
    old=json.loads(p.read_text());new=replace(old)
    if p.name=='frame_art_music_catalog.json':
        e=new['entries'][KEY+'__3840x2160.jpg'];assert e['content_id']==NEW
        e.update(source_hash=recipe['final_jpeg_sha256'],updated_at=stamp,canonical_key=KEY+'__3840x2160.jpg')
    if new!=old:atomic(p,new)
recipe.update(deployed=True,deployed_content_id=NEW,deployed_at=stamp)
atomic(music/'recipes'/(KEY+'.json'),recipe)
print('Committed',KEY,NEW)
""".replace('RUN', repr(REMOTE)).replace('KEY', repr(entry['key'])).replace('OLD', repr(old_id)).replace('NEW', repr(new_id))
    print(remote(script), flush=True)


def deploy():
    plan_path = OUT / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else prepare()
    catalog = get_remote_catalog([e['key'] for e in plan['entries']])
    replace, old_ids = initialize_remote(plan, catalog)
    existing = inventory()
    if old_ids & existing:
        art = connect_art()
        try:
            current = art.get_current().get('content_id')
            if current in old_ids:
                safe = next(cid for cid in existing if cid not in old_ids)
                art.select_image(safe, show=False)
        finally:
            art.close()
        time.sleep(8)
    journal_path = OUT / 'journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {'entries': {}}
    for position, entry in enumerate(replace, 1):
        state = journal['entries'].setdefault(entry['key'], {})
        old_id = catalog[entry['key']]['content_id']
        if state.get('status') == 'committed':
            continue
        before = set(state.get('inventory_before') or inventory())
        state.update(old_id=old_id, inventory_before=sorted(before), status=state.get('status', 'ready'))
        atomic_json(journal_path, journal)
        if old_id in inventory():
            print(f'[{position}/{len(replace)}] deleting {old_id} {entry["artist"]} — {entry["title"]}', flush=True)
            mutate('delete', old_id=old_id)
            time.sleep(8)
        if old_id in inventory():
            raise RuntimeError(f'TV still contains deleted ID {old_id}')
        if state.get('status') not in {'uploading', 'uploaded'}:
            state['status'] = 'deleted'
            atomic_json(journal_path, journal)
        new_id = state.get('new_id')
        if state.get('status') == 'uploading' and not new_id:
            baseline = set(state['upload_baseline'])
            added = inventory() - baseline
            if len(added) == 1:
                new_id = added.pop()
                state.update(status='uploaded', new_id=new_id)
                atomic_json(journal_path, journal)
            elif len(added) > 1:
                raise RuntimeError(
                    f'Cannot safely recover interrupted upload for {entry["key"]}: '
                    f'unexpected TV IDs {sorted(added)}'
                )
        if not new_id or new_id not in inventory():
            baseline = inventory()
            state.update(status='uploading', upload_baseline=sorted(baseline))
            atomic_json(journal_path, journal)
            print(f'[{position}/{len(replace)}] uploading {entry["key"]}', flush=True)
            new_id = mutate('upload', image=Path(entry['folder'])/'compressed.jpg', before=baseline)
            if not isinstance(new_id, str) or not new_id.startswith('MY_F'):
                raise RuntimeError(f'Invalid new TV ID: {new_id!r}')
            state.update(status='uploaded', new_id=new_id);atomic_json(journal_path, journal)
            time.sleep(8)
        after = inventory()
        if new_id not in after or old_id in after or not before - {old_id} <= after:
            raise RuntimeError(f'Inventory verification failed for {entry["key"]}')
        commit_remote(entry, old_id, new_id)
        state.update(status='committed', verified_inventory=True);atomic_json(journal_path, journal)
    remote("import pathlib,json\np=pathlib.Path("+repr(CONTROL)+")\nassert json.loads(p.read_text())['run']=="+repr(REMOTE)+"\np.unlink()")
    subprocess.run(command() + ['ha apps start ' + APP], check=True, timeout=60)
    journal['complete'] = True;atomic_json(journal_path, journal)
    print('DEPLOYMENT COMPLETE', len(replace), 'replacements;', len(plan['entries'])-len(replace), 'retained', flush=True)


if __name__ == '__main__':
    deploy()
