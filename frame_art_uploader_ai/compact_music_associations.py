#!/usr/bin/env python3
import argparse
import json
import sys
import types
from pathlib import Path

# Allow running this utility outside the full add-on runtime.
if "samsungtvws" not in sys.modules:
    samsungtvws = types.ModuleType("samsungtvws")
    samsungtvws.SamsungTVWS = object
    sys.modules["samsungtvws"] = samsungtvws
if "PIL" not in sys.modules:
    pil = types.ModuleType("PIL")
    pil.Image = object()
    sys.modules["PIL"] = pil
if "cover_art" not in sys.modules:
    cover = types.ModuleType("cover_art")
    cover.SOURCE_DIR = Path(".")
    cover.BACKGROUND_DIR = Path(".")
    cover.COMPRESSED_DIR = Path(".")
    cover.WIDESCREEN_DIR = Path(".")
    cover.JPEG_MAX_BYTES = 4 * 1024 * 1024
    cover.download_artwork = lambda *a, **k: None
    cover.compress_png_path_to_jpeg_max_bytes = lambda *a, **k: (True, 0)
    cover.ensure_dirs = lambda *a, **k: None
    cover.itunes_lookup = lambda *a, **k: {}
    cover.itunes_search = lambda *a, **k: {}
    cover.normalize_key = lambda *a, **k: "k"
    cover.generate_reference_frame_from_album = lambda *a, **k: (b"", b"", None, None)
    cover.generate_local_fallback_frame_from_album = lambda *a, **k: (b"", b"")
    cover.resolve_artwork_url = lambda *a, **k: ""
    sys.modules["cover_art"] = cover

import uploader


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact music associations to canonical keys.")
    parser.add_argument(
        "--path",
        default=str(uploader.MUSIC_ASSOCIATIONS_PATH),
        help="Path to music associations JSON",
    )
    parser.add_argument(
        "--session-ttl-days",
        type=int,
        default=uploader.MUSIC_ASSOCIATION_SESSION_TTL_DAYS,
        help="Deprecated and ignored; session/shazam aliases no longer expire.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write compacted output back to disk. Without this flag, only prints stats.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    catalog = uploader.load_frame_art_catalog(path)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    compacted, stats = uploader.compact_music_association_entries(entries, session_ttl_days=args.session_ttl_days)

    output = {
        "path": str(path),
        "apply": bool(args.apply),
        **stats,
    }
    print(json.dumps(output, indent=2))

    if args.apply:
        catalog["entries"] = compacted
        uploader.persist_frame_art_catalog(path, catalog)
        print(json.dumps({"written": str(path), "entries": len(compacted)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
