"""Delete-first import of a frozen reviewed release. No image generation calls.

A persistent control file holds ordinary uploader requests while this worker owns
its lock. An unacknowledged upload always pauses for reconciliation, never replay.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

CONTROL = Path('/share/frame_art_migration/active.json')
MUSIC = Path('/media/frame_ai/music')
SHARE = Path('/share')
DATA = Path('/data')
META = ('frame_art_music_catalog.json', 'frame_art_music_associations.json',
        'frame_art_music_overrides.json', 'frame_art_music_triage.json',
        'frame_art_ambient_catalog.json', 'frame_art_holidays_catalog.json',
        'frame_art_uploader_last.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def active_map(catalog):
    return {k: v.get('content_id') for k, v in catalog.get('entries', {}).items()
            if v.get('state', 'active') == 'active' and v.get('content_id')}


def checked_file(root, item):
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != item['sha256']:
        raise ValueError(f'Release checksum/path mismatch: {item["path"]}')
    return path


class Paused(RuntimeError):
    pass


class Migration:
    def __init__(self, control, art, *, music=MUSIC, share=SHARE, data=DATA, sleep=time.sleep):
        self.control = Path(control)
        self.config = read(control)
        self.root = Path(self.config['release'])
        self.release = read(self.root / 'release.json')
        self.run = Path(self.config['run'])
        self.run.mkdir(parents=True, exist_ok=True)
        self.music, self.share, self.data = Path(music), Path(share), Path(data)
        self.art, self.sleep = art, sleep
        self.journal = read(self.run / 'journal.json', {'release_hash': sha(self.root / 'release.json'),
            'rows': {}, 'phase': 'preflight', 'created_at': now(), 'last_mutation': 0})
        if self.journal['release_hash'] != sha(self.root / 'release.json'):
            raise ValueError('Frozen release changed after migration started')

    def save(self, **fields):
        self.journal.update(fields, updated_at=now())
        write(self.run / 'journal.json', self.journal)
        rows = self.journal['rows']
        write(self.run / 'status.json', dict(phase=self.journal['phase'], updated_at=now(),
            completed=sum(r.get('state') == 'complete' for r in rows.values()),
            total=len(self.release['entries']), current=self.journal.get('current'), error=self.journal.get('error')))

    def check_pause(self):
        if read(self.control, {}).get('paused', True):
            raise Paused('Migration paused by control file')

    def inventory(self):
        raw = self.art.available()
        if not isinstance(raw, list) or any(not isinstance(v, dict) or not v.get('content_id') for v in raw):
            raise ValueError('TV did not return a valid inventory')
        return {v['content_id'] for v in raw}

    def mutate(self, fn, *args, **kwargs):
        self.check_pause()
        delay = max(0, 10 - (time.time() - self.journal['last_mutation']))
        self.sleep(delay)
        self.journal['last_mutation'] = time.time()
        self.save()
        return fn(*args, **kwargs)

    def backup(self):
        if self.journal.get('backup_verified'):
            return
        self.save(phase='backup')
        backup = self.run / 'backup'
        backup.mkdir(exist_ok=True)
        pairs = [(self.music, backup / 'music')]
        pairs += [(self.share / name, backup / 'share' / name) for name in META if (self.share / name).exists()]
        pairs += [(p, backup / 'data' / p.name) for p in self.data.glob('frame_art*json')]
        # No credentials/options/tokens are copied into this rollback package.
        checksums = {}
        for source, target in pairs:
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
                files = [(p, target / p.relative_to(source)) for p in source.rglob('*') if p.is_file()]
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files = [(source, target)]
            for original, copied in files:
                digest = sha(original)
                if digest != sha(copied):
                    raise ValueError(f'Backup verification failed: {original.name}')
                checksums[str(copied.relative_to(backup))] = digest
        write(backup / 'checksums.json', checksums)
        self.save(backup_verified=True)

    def preflight(self):
        for row in self.release['entries']:
            for item in row['files'].values():
                checked_file(self.root, item)
            with Image.open(checked_file(self.root, row['files']['compressed'])) as im:
                if im.format != 'JPEG' or im.size != (3840,2160) or im.mode != 'RGB':
                    raise ValueError('Invalid prepared TV image')
                im.verify()
            if (self.root / row['files']['compressed']['path']).stat().st_size > 4*1024**2:
                raise ValueError('Prepared TV image exceeds 4 MiB')
        if not self.journal.get('backup_verified'):
            catalog = read(self.share / META[0])
            if active_map(catalog) != self.release['expected_catalog']:
                raise ValueError('Live music catalog changed; rebuild the release plan before deleting')
            if shutil.disk_usage(self.run).free < 5*1024**3:
                raise ValueError('Insufficient free space for verified rollback package')
        self.backup()
        ids = self.inventory()
        protected = set()
        for name in META[4:6]:
            protected.update(active_map(read(self.share / name, {})).values())
        old_ids = {cid for row in self.release['entries'] for cid in row['old_ids']}
        reused = {row['reuse_id'] for row in self.release['entries'] if row.get('reuse_id')}
        if old_ids & (protected | reused):
            raise ValueError('Retired IDs overlap retained or other-category artwork')
        if not self.journal.get('inventory_verified'):
            write(self.run / 'initial-inventory.json', sorted(ids))
            write(self.run / 'initial-current.json', self.art.get_current())
            self.save(inventory_verified=True)
        self.save(phase='replacing', error=None)

    def switch_from(self, old_id, ids):
        current = self.art.get_current()
        current_id = current.get('content_id') if isinstance(current, dict) else None
        if not current_id:
            raise ValueError('Cannot identify currently selected TV art')
        if current_id != old_id:
            return
        retained = {row['reuse_id'] for row in self.release['entries'] if row.get('reuse_id')}
        retained.update(r['content_id'] for r in self.journal['rows'].values() if r.get('state') == 'complete')
        choices = sorted((retained & ids) - {old_id})
        if not choices:
            raise ValueError('No verified retained artwork to select before deletion')
        self.mutate(self.art.select_image, choices[0], show=False)
        if self.art.get_current().get('content_id') == old_id:
            raise ValueError('TV has not switched away from artwork being replaced')

    def replace(self, row):
        key = str(row['album_id'])
        state = self.journal['rows'].setdefault(key, {'state': 'prepared', 'deleted': []})
        self.save(current={'album_id': row['album_id'], 'artist': row['artist'], 'album': row['title']})
        if state['state'] == 'complete':
            return
        ids = self.inventory()
        if state['state'] == 'upload_started':
            # Even zero new IDs is not proof that a delayed upload cannot finish.
            state['uncertain_new_ids'] = sorted(ids - set(state['before_upload']))
            self.save()
            raise Paused(f'Upload acknowledgement uncertain for album {key}; reconcile inventory before resuming')
        if state['state'] not in ('uploaded', 'verified'):
            checksums = read(self.run / 'backup/checksums.json')
            for cid in row['old_ids']:
                if cid in state['deleted']:
                    continue
                old = row['old_files'][cid]
                path = self.run / 'backup/music/widescreen-compressed' / old
                if sha(path) != checksums['music/widescreen-compressed/' + old]:
                    raise ValueError('Rollback image changed or is missing')
                if cid in ids:
                    self.switch_from(cid, ids)
                    state['state'], state['deleting_id'] = 'deleting', cid
                    self.save()
                    self.mutate(self.art.delete, cid)
                    ids = self.inventory()
                    if cid in ids:
                        raise ValueError(f'Deletion not confirmed: {cid}')
                state['deleted'].append(cid)
                self.save()
            reuse = row.get('reuse_id')
            if reuse and reuse in ids:
                state.update(state='uploaded', content_id=reuse, reused=True)
                self.save()
            else:
                image = checked_file(self.root, row['files']['compressed']).read_bytes()
                state.update(state='upload_started', before_upload=sorted(ids))
                self.save()
                cid = self.mutate(self.art.upload, image, file_type='JPEG', matte='none')
                if not isinstance(cid, str) or not cid.startswith('MY_F') or cid in ids:
                    raise ValueError('Missing or invalid upload acknowledgement; reconcile inventory')
                state.update(state='uploaded', content_id=cid, reused=False)
                self.save()
        cid = state['content_id']
        if cid not in self.inventory():
            raise ValueError(f'Uploaded/retained image is missing from TV: {cid}')
        if not state.get('reused'):
            thumbnail = self.art.get_thumbnail(cid)
            if not isinstance(thumbnail, (bytes, bytearray)):
                raise ValueError('TV thumbnail response is not image bytes')
            with Image.open(BytesIO(thumbnail)) as image:
                image.verify()
            (self.run / 'thumbnails').mkdir(exist_ok=True)
            (self.run / 'thumbnails' / f'{key}.jpg').write_bytes(thumbnail)
        state['state'] = 'verified'
        self.save()
        self.commit_row(row, state)
        state['state'] = 'complete'
        self.save()

    def commit_row(self, row, state):
        for kind, item in row['files'].items():
            source = checked_file(self.root, item)
            target = self.music / item['target']
            if not target.resolve().is_relative_to(self.music.resolve()):
                raise ValueError('Invalid media destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha(target) != item['sha256']:
                temp = target.with_name(target.name + '.migration-tmp')
                shutil.copy2(source, temp)
                with temp.open('rb') as stream:
                    os.fsync(stream.fileno())
                temp.replace(target)
        catalog = read(self.share / META[0])
        for key, value in catalog['entries'].items():
            if value.get('content_id') in row['old_ids']:
                value.update(content_id='', state='deleted', deleted_at=now(), delete_reason='reviewed_release')
        catalog['entries'][row['catalog_key']] = dict(content_id=state['content_id'], state='active',
            source_hash=row['files']['compressed']['sha256'], canonical_key=row['catalog_key'], updated_at=now())
        write(self.share / META[0], catalog)

    def finalize(self):
        self.save(phase='reconciling')
        ids = self.inventory()
        states = self.journal['rows']
        expected_ids = {states[str(r['album_id'])]['content_id'] for r in self.release['entries']}
        retired = {cid for r in self.release['entries'] for cid in r['old_ids']}
        if not expected_ids <= ids or retired & ids:
            raise ValueError('Final TV inventory does not match the completed migration')
        initial = set(read(self.run / 'initial-inventory.json'))
        if (initial - retired) - ids:
            raise ValueError('Unrelated/retained TV artwork changed during migration')
        metadata = reconcile(self.release, states, self.run / 'backup', self.music, self.share)
        # Deterministic reconstruction from frozen metadata is safe to repeat if any
        # individual atomic replace is interrupted. Maintenance stays held throughout.
        for target, document in metadata.items():
            write(Path(target), document)
        wanted = {item['target'] for row in self.release['entries'] for item in row['files'].values()}
        for directory in ('source', 'background', 'widescreen', 'widescreen-compressed'):
            for path in (self.music / directory).rglob('*'):
                relative = path.relative_to(self.music).as_posix()
                if path.is_file() and relative not in wanted:
                    saved = self.run / 'backup/music' / relative
                    if not saved.exists() or sha(saved) != sha(path):
                        raise ValueError(f'Unexpected extra file; preserve and reconcile: {path.name}')
                    path.unlink()
        for row in self.release['entries']:
            if sha(self.music / row['files']['compressed']['target']) != row['files']['compressed']['sha256']:
                raise ValueError('Final media hash verification failed')
        mapping = {cid: states[str(row['album_id'])]['content_id'] for row in self.release['entries'] for cid in row['old_ids']}
        current_before = read(self.run / 'initial-current.json', {}).get('content_id')
        if current_before in mapping:
            self.mutate(self.art.select_image, mapping[current_before], show=False)
        for source in (self.run / 'backup/data').glob('*.json'):
            write(self.data / source.name, replace_ids(read(source), mapping))
        display = self.share / 'frame_art_display/current.json'
        if display.exists():
            write(display, replace_ids(read(display), mapping))
        self.save(phase='complete', completed_at=now(), current=None)
        self.control.unlink()

    def execute(self):
        self.check_pause()
        self.preflight()
        count = 0
        for row in self.release['entries']:
            self.check_pause()
            if self.journal['rows'].get(str(row['album_id']), {}).get('state') == 'complete':
                continue
            self.replace(row)
            count += 1
            # One-album canary, followed by batches of five. Only one replacement
            # is ever outstanding because deletion and upload are serial per album.
            if count == 1 or (count-1) % 5 == 0:
                self.sleep(60)
        self.finalize()


def replace_ids(value, mapping):
    if isinstance(value, dict):
        return {k: replace_ids(v, mapping) for k,v in value.items()}
    if isinstance(value, list):
        return [replace_ids(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def reconcile(release, states, backup, music, share):
    stamp = now()
    catalog, index, manifest, associations = {}, {}, {}, {}
    old_associations = read(backup / 'share/frame_art_music_associations.json', {}).get('entries', {})
    old_to_row, key_to_row = {}, {}
    for row in release['entries']:
        for cid in row['old_ids'] + ([row['reuse_id']] if row.get('reuse_id') else []):
            old_to_row[cid] = row
        for key in row['lookup_keys']:
            key_to_row[key] = row
    def record(row):
        return dict(cache_key=row['key'], catalog_key=row['catalog_key'],
            content_id=states[str(row['album_id'])]['content_id'], artist=row['artist'], album=row['title'],
            collection_id=row['collection_id'], verified=True, source_quality='trusted_cache',
            cache_reuse_recommended=True, cache_reuse_confidence=1.0, match_confidence=1.0,
            match_source='manual_review', updated_at=stamp)
    for key, old in old_associations.items():
        if not isinstance(old, dict):
            continue
        row = old_to_row.get(old.get('content_id')) or key_to_row.get(old.get('cache_key'))
        if row and not (key.startswith('cache::') and key[7:] in release['wrong_collection_keys']):
            associations[key] = record(row)
    for row in release['entries']:
        rec = record(row)
        paths = {kind: str(music / item['target']) for kind,item in row['files'].items()}
        entry = dict(text_key=f'{row["artist"]} — {row["title"]}', collection_id=row['collection_id'],
            collectionId=row['collection_id'], content_id=rec['content_id'], status='ok',
            output_path=paths['compressed'], compressed_output_path=paths['compressed'],
            source_path=paths['source'], background_output_path=paths.get('background', ''),
            widescreen_output_path=paths.get('widescreen', ''), recipe_path=paths['recipe'],
            updated_at=stamp, reviewed_asset=row['asset'], pipeline='reviewed_release',
            prompt_variant=row['recipe'].get('mode', row['recipe'].get('pipeline', 'legacy_original')),
            model_used=row['recipe'].get('model', row['recipe'].get('model_used')), render_style=row['finish'])
        catalog[row['catalog_key']] = dict(content_id=rec['content_id'], state='active', updated_at=stamp,
            source_hash=row['files']['compressed']['sha256'], canonical_key=row['catalog_key'])
        for key in row['lookup_keys']:
            index[key] = dict(entry, canonical_key=row['key'])
            associations['cache::' + key] = rec
        for text in row['text_keys']:
            manifest[text] = entry
            norm = re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()
            associations['album_norm::' + norm] = rec
        primary_artist = re.split(r"\s*(?:,|&|\band\b|\bwith\b|\bfeat\.?\b|\bfeaturing\b|\bx\b)\s*", row['artist'], flags=re.I)[0]
        norm = re.sub(r'[^a-z0-9]+', ' ', f"{primary_artist} {row['title']}".lower()).strip()
        associations['album_norm::' + norm] = rec
    # Historical overrides may point at a removed filename. Confirmed choices now
    # take precedence for every reviewed album; unrelated overrides are preserved.
    overrides = read(backup / 'share/frame_art_music_overrides.json', {'entries': {}})
    for key, rec in associations.items():
        if key.startswith('album_norm::'):
            overrides.setdefault('entries', {})[key] = dict(artist=rec['artist'], album=rec['album'],
                catalog_key=rec['catalog_key'], reason='confirmed_review', updated_at=stamp)
    wrap = lambda entries: dict(version=1, updated_at=stamp, entries=entries)
    return {str(share / META[0]): wrap(catalog), str(share / META[1]): wrap(associations),
        str(share / META[2]): overrides, str(music / 'index.json'): wrap(index),
        str(music / 'manifest.json'): dict(generatedAt=stamp, entries=manifest)}


def main():
    import uploader
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, default=CONTROL)
    args = parser.parse_args()
    config = read(args.control, {})
    if not config or config.get('paused', True):
        return
    import socket
    socket.setdefaulttimeout(30)
    uploader.RUNTIME_OPTIONS = uploader.load_options()
    with uploader.worker_lock() as acquired:
        if not acquired:
            return
        worker = None
        try:
            art = uploader.create_art_client(uploader.create_tv_client(uploader.RUNTIME_OPTIONS['tv_ip']))
            if config.get('probe_only'):
                inventory = art.available()
                if not isinstance(inventory, list):
                    raise ValueError('TV did not return inventory')
                write(Path(config['run']) / 'connection-probe.json',
                    dict(checked_at=now(), inventory=inventory, current=art.get_current()))
                config.update(paused=True, probe_only=False, probe_ok=True)
                write(args.control, config)
            else:
                worker = Migration(args.control, art)
                worker.execute()
        except Exception as exc:
            config = read(args.control, config)
            config.update(paused=True, error=str(exc), paused_at=now())
            write(args.control, config)
            if worker:
                worker.save(phase='paused', error=str(exc))
            print(f'Migration paused: {exc}', flush=True)
        finally:
            uploader.close_registered_tv_connections(context="migration")


if __name__ == '__main__':
    main()
