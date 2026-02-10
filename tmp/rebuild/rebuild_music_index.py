import json
from datetime import datetime, timezone
from pathlib import Path

INDEX_PATH = Path('/Users/jsands/index.json')
CATALOG_PATH = Path('/Users/jsands/frame_art_music_catalog.json')
ASSOCIATIONS_PATH = Path('/Users/jsands/frame_art_music_associations.json')
MANIFEST_PATH = Path('/Users/jsands/manifest.json')
OUT_PATH = Path('/Users/jsands/Code/frame-art-addon/tmp/rebuild/index.rebuilt.json')


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected object JSON in {path}')
    return data


def normalize_text(s: str) -> str:
    return ' '.join(str(s or '').strip().lower().split())


def text_key_from_assoc(record: dict) -> str:
    artist = str(record.get('artist', '')).strip()
    album = str(record.get('album', '')).strip()
    if artist and album:
        return f"{normalize_text(artist)} — {normalize_text(album)}"
    return normalize_text(artist or album)


def entry_key_from_catalog_key(catalog_key: str) -> str:
    stem = Path(catalog_key).name.rsplit('.', 1)[0]
    stem = stem.replace('__3840x2160__background', '')
    stem = stem.replace('__3840x2160', '')
    return stem


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()

    index = load_json(INDEX_PATH)
    catalog = load_json(CATALOG_PATH)
    associations = load_json(ASSOCIATIONS_PATH)
    manifest = load_json(MANIFEST_PATH)

    existing_entries = index.get('entries') if isinstance(index.get('entries'), dict) else {}
    catalog_entries = catalog.get('entries') if isinstance(catalog.get('entries'), dict) else {}
    assoc_entries = associations.get('entries') if isinstance(associations.get('entries'), dict) else {}
    manifest_entries = manifest.get('entries') if isinstance(manifest.get('entries'), dict) else {}

    manifest_by_collection_id: dict[str, dict] = {}
    manifest_by_text_key: dict[str, dict] = {}
    for _, item in manifest_entries.items():
        if not isinstance(item, dict):
            continue
        cid = item.get('collection_id')
        if cid is None:
            cid = item.get('collectionId')
        if cid is not None:
            manifest_by_collection_id[str(cid)] = item
        text_key = normalize_text(str(item.get('text_key', '')).replace('—', '-').replace('-', ' — ', 1))
        if text_key:
            manifest_by_text_key[text_key] = item

    assoc_by_catalog_key: dict[str, dict] = {}
    for _, rec in assoc_entries.items():
        if not isinstance(rec, dict):
            continue
        k = str(rec.get('catalog_key', '')).strip()
        if k:
            assoc_by_catalog_key[k] = rec

    rebuilt_entries: dict[str, dict] = {}

    for catalog_key, cat_item in catalog_entries.items():
        if not isinstance(cat_item, dict):
            continue
        catalog_key = str(catalog_key)
        entry_key = entry_key_from_catalog_key(catalog_key)
        existing = existing_entries.get(entry_key) if isinstance(existing_entries.get(entry_key), dict) else {}

        man_item = manifest_by_collection_id.get(entry_key)
        assoc_item = assoc_by_catalog_key.get(catalog_key)

        text_key = ''
        if existing:
            text_key = str(existing.get('text_key', '')).strip()
        if not text_key and isinstance(man_item, dict):
            text_key = normalize_text(str(man_item.get('text_key', '')).replace(' - ', ' — ').replace('—', ' — '))
        if not text_key and isinstance(assoc_item, dict):
            text_key = text_key_from_assoc(assoc_item)

        content_id = str(cat_item.get('content_id', '')).strip() or str(existing.get('content_id', '')).strip()

        runtime_path = f"/media/frame_ai/music/widescreen-compressed/{catalog_key}"
        output_path = str(existing.get('output_path', '')).strip() or runtime_path
        compressed_output_path = str(existing.get('compressed_output_path', '')).strip() or runtime_path

        payload = dict(existing)
        if text_key:
            payload['text_key'] = text_key
        payload['status'] = str(payload.get('status', 'ok')).strip() or 'ok'
        if content_id:
            payload['content_id'] = content_id
        payload['prompt_variant'] = str(payload.get('prompt_variant', 'reference_background_nomask')).strip() or 'reference_background_nomask'
        payload['output_path'] = output_path
        payload['compressed_output_path'] = compressed_output_path
        payload['updated_at'] = str(payload.get('updated_at', '')).strip() or str(cat_item.get('updated_at', '')).strip() or now

        if isinstance(man_item, dict):
            req = str(man_item.get('request_id', '')).strip()
            if req and not str(payload.get('request_id', '')).strip():
                payload['request_id'] = req

        rebuilt_entries[entry_key] = payload

    # Keep any existing entries not present in catalog to avoid accidental loss.
    for key, value in existing_entries.items():
        if key not in rebuilt_entries and isinstance(value, dict):
            rebuilt_entries[key] = value

    out = {
        'version': 1,
        'updated_at': now,
        'entries': dict(sorted(rebuilt_entries.items(), key=lambda kv: kv[0])),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding='utf-8')

    with_text_key = sum(1 for v in rebuilt_entries.values() if isinstance(v, dict) and str(v.get('text_key', '')).strip())
    print(json.dumps({
        'rebuilt_entries': len(rebuilt_entries),
        'catalog_entries': len(catalog_entries),
        'existing_index_entries': len(existing_entries),
        'entries_with_text_key': with_text_key,
        'output': str(OUT_PATH),
    }, indent=2))


if __name__ == '__main__':
    main()
