"""Freeze confirmed selections and layer hashes for the TV migration. Local only."""
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import curation
from store import ROOT, db


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    release = ROOT / 'release-original-finish'
    release.mkdir(exist_ok=True)
    audit = ROOT / 'reports/cache-rebuild-audit'
    previous = json.loads((audit / 'replacement-plan.json').read_text())
    confirmed = curation.reconciliation()
    if not confirmed['review_complete'] or confirmed['render_style']['shadow'] != 'original' or confirmed['render_style']['feather'] != 0:
        raise ValueError('Review/finish is not ready')
    with db() as c:
        albums = {str(a['id']): dict(a) for a in c.execute('SELECT * FROM albums')}
        candidates = {str(a['id']): dict(a) for a in c.execute('SELECT * FROM candidates')}
    catalog = json.loads((audit / 'share/frame_art_music_catalog.json').read_text())
    expected = {k: v['content_id'] for k,v in catalog['entries'].items() if v.get('state','active') == 'active' and v.get('content_id')}
    background_audit = {a['album_id']: a for a in json.loads((ROOT / 'reports/original-finish-audit.json').read_text())}
    old_hashes = {}
    for key,cid in expected.items():
        path = ROOT / 'snapshot/music/widescreen-compressed' / key
        if path.exists():
            digest = sha(path)
            if catalog['entries'][key].get('source_hash') == digest:
                old_hashes.setdefault(digest, []).append((key,cid))
    wrong = [key for key,a in confirmed['entries'].items() if key.isdigit() and a['collection_id'] and int(key) != int(a['collection_id'])]
    rows = []
    items = list(confirmed['entries'].items())
    def build(pair):
        old_key, a = pair
        asset = a['selection']['asset']
        kind, selected_id = asset.split(':')
        key = str(a['collection_id']) if old_key in wrong else old_key
        folder = release / key
        folder.mkdir(exist_ok=True)
        files = {}
        def add(kind, source, target):
            source = Path(source)
            destination = folder / (kind + source.suffix)
            digest = sha(source)
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            elif sha(destination) != digest:
                raise ValueError('Frozen release file changed; build a new release')
            files[kind] = dict(path=str(destination.relative_to(release)), sha256=digest, target=target)
        compressed = curation.render_asset(asset, 'original', 0)
        add('compressed', compressed, f'widescreen-compressed/{key}__3840x2160.jpg')
        if kind == 'candidate':
            cf = ROOT / candidates[selected_id]['folder']
            recipe = json.loads((cf / 'recipe.json').read_text())
            source = cf / 'source/cover.png'
            background = cf / 'background/generated.png'
            wide = compressed.parent.parent / 'widescreen' / (compressed.stem + '.png')
        else:
            selected = albums[selected_id]
            source = ROOT / selected['source_path']
            original_key = Path(selected['current_path']).stem
            matches = background_audit.get(a['album_id'], {}).get('background_paths', [])
            background = Path(matches[0]) if matches else None
            recipe = dict(mode='saved_original', model_used=json.loads(selected['metadata']).get('legacy',{}).get('model_used'),
                saved_original_path=selected['current_path'], saved_original_sha256=sha(compressed), finish='baked_original')
            wide = folder / 'saved-original.png'
            if not wide.exists():
                with Image.open(compressed) as im:
                    im.convert('RGB').save(wide, compress_level=1)
        if background:
            add('background', background, f'background/{key}__3840x2160__background.png')
        add('source', source, f'source/{key}__source{source.suffix}')
        add('widescreen', wide, f'widescreen/{key}__3840x2160.png')
        recipe.update(reviewed_asset=asset, final_shadow='original', final_feather_px=0,
            final_source_sha256=sha(source), final_jpeg_sha256=sha(compressed))
        recipe_path = folder / 'approved-recipe.json'
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise ValueError('Frozen recipe changed')
        recipe_path.write_text(json.dumps(recipe, indent=2))
        add('recipe', recipe_path, f'recipes/{key}.json')
        lookup_keys = [key] + [x for x in a['aliases'] if x not in wrong and x != key]
        text_keys = {f'{a["artist"]} — {a["title"]}'}
        for original_key in [old_key] + a['aliases']:
            meta = confirmed['original_metadata_by_key'][original_key]
            for section in ['legacy','index']:
                if meta.get(section, {}).get('text_key'):
                    text_keys.add(meta[section]['text_key'])
        reuse = next(iter(old_hashes.get(files['compressed']['sha256'], [])), (None,None))
        return dict(album_id=a['album_id'], key=key, old_key=old_key, lookup_keys=lookup_keys, text_keys=sorted(text_keys),
            artist=a['artist'], title=a['title'], collection_id=a['collection_id'], asset=asset, files=files,
            recipe=recipe, finish={'shadow':'original','feather':0}, catalog_key=f'{key}__3840x2160.jpg',
            reuse_id=reuse[1], old_ids=[], old_files={})
    # Image rendering is local CPU work, with bounded memory use.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(build, items):
            rows.append(row)
            print(f'Prepared {len(rows)}/{len(items)}: {row["artist"]} — {row["title"]}', flush=True)
    retained = {r['reuse_id'] for r in rows if r['reuse_id']}
    by_oldkey = {r['old_key']: r for r in rows}
    owners = {}
    for filename,cid in expected.items():
        old_key = filename.rsplit('__3840x2160',1)[0]
        canonical = confirmed['alias_map'].get(old_key)
        if not canonical:
            raise ValueError(f'Unmapped old catalog entry: {filename}')
        if cid in retained:
            continue
        row = by_oldkey[canonical]
        if cid in owners and owners[cid] != row['album_id']:
            raise ValueError('An old TV image is shared by unrelated albums')
        owners[cid] = row['album_id']
        if cid not in row['old_ids']:
            row['old_ids'].append(cid)
            row['old_files'][cid] = filename
    frozen = dict(format='frame-art-release-v1', finish={'shadow':'original','feather':0},
        entries=rows, expected_catalog=expected, wrong_collection_keys=wrong,
        review_choices={k: a['selection']['asset'] for k,a in confirmed['entries'].items()},
        summary=dict(albums=len(rows), retained=len(retained), uploads=sum(not r['reuse_id'] for r in rows),
            retired=len(owners), layer_bytes=sum((release / f['path']).stat().st_size for r in rows for f in r['files'].values())))
    (release / 'release.json').write_text(json.dumps(frozen,indent=2))
    (release / 'confirmed-reconciliation.json').write_text(json.dumps(confirmed,indent=2))
    print(json.dumps(frozen['summary']), flush=True)


if __name__ == '__main__':
    main()
