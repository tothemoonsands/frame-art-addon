import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from samsungtvws import SamsungTVWS

from cover_art import (
    BACKGROUND_DIR,
    COMPRESSED_DIR,
    JPEG_MAX_BYTES,
    SOURCE_DIR,
    WIDESCREEN_DIR,
    compress_png_path_to_jpeg_max_bytes,
    download_artwork,
    ensure_dirs,
    generate_local_fallback_frame_from_album,
    itunes_lookup,
    itunes_search,
    normalize_key,
    generate_reference_frame_from_album,
    resolve_artwork_url,
)

OPTIONS_PATH = "/data/options.json"
STATUS_PATH = "/share/frame_art_uploader_last.json"
STATE_PATH = "/data/frame_art_uploader_state.json"

# One-shot restore request written by Home Assistant.
RESTORE_REQUEST_PATH = "/share/frame_art_restore_request.json"
RESTORE_QUEUE_DIR = Path("/share/frame_art_restore_queue")
WORKER_LOCK_PATH = Path("/share/frame_art_uploader_worker.lock")
FALLBACK_DIR = Path("/media/frame_ai/music/fallback")
HOLIDAY_CATALOG_PATH = Path("/share/frame_art_holidays_catalog.json")
AMBIENT_CATALOG_PATH = Path("/share/frame_art_ambient_catalog.json")
MUSIC_CATALOG_PATH = Path("/share/frame_art_music_catalog.json")
MUSIC_ASSOCIATIONS_PATH = Path("/share/frame_art_music_associations.json")
MUSIC_ERRORS_PATH = Path("/share/frame_art_music_errors.json")

MYF_RE = re.compile(r"^MY[_-]F(\d+)", re.IGNORECASE)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AMBIENT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PHASE_SALT = {
    "pre_dawn": 101,
    "midday": 303,
    "evening": 505,
    "night": 707,
}
UNKNOWN_PHASE = "night"

RUNTIME_OPTIONS: dict[str, Any] = {}
ADDON_VERSION = "0.3.6"
HOLIDAY_ALIASES = {
    "football": "huskers",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def write_status(payload: dict) -> None:
    payload["ts"] = time.monotonic()
    atomic_write_json(Path(STATUS_PATH), payload)
    log_event("status", **payload)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path = path.parent / tmp_name
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_options() -> dict:
    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_local_uploaded_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            cid = item.strip()
            if cid:
                out.append(cid)
    return out


def append_local_uploaded_id(state: dict, target_cid: Any) -> None:
    existing = sanitize_local_uploaded_ids(state.get("local_uploaded_ids"))
    state["local_uploaded_ids"] = existing
    if isinstance(target_cid, str):
        cid = target_cid.strip()
        if cid:
            state["local_uploaded_ids"].append(cid)


def append_uploaded_id(state: dict, state_key: str, target_cid: Any) -> None:
    existing = sanitize_local_uploaded_ids(state.get(state_key))
    state[state_key] = existing
    if isinstance(target_cid, str):
        cid = target_cid.strip()
        if cid:
            state[state_key].append(cid)


def load_state() -> dict:
    p = Path(STATE_PATH)
    if not p.exists():
        return {
            "last_applied": None,
            "local_uploaded_ids": [],
            "cover_uploaded_ids": [],
            "pending_keep_count_cleanup": False,
        }
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_applied": None,
            "local_uploaded_ids": [],
            "cover_uploaded_ids": [],
            "pending_keep_count_cleanup": False,
        }
    if not isinstance(data, dict):
        return {
            "last_applied": None,
            "local_uploaded_ids": [],
            "cover_uploaded_ids": [],
            "pending_keep_count_cleanup": False,
        }
    data.setdefault("last_applied", None)
    data["local_uploaded_ids"] = sanitize_local_uploaded_ids(data.get("local_uploaded_ids"))
    data["cover_uploaded_ids"] = sanitize_local_uploaded_ids(data.get("cover_uploaded_ids"))
    data["pending_keep_count_cleanup"] = bool(data.get("pending_keep_count_cleanup", False))
    return data


def save_state(state: dict) -> None:
    Path(STATE_PATH).write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return None


def guess_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".png":
        return "PNG"
    return "PNG"


def newest_candidate(inbox_dir: str, prefix: str) -> Optional[Path]:
    d = Path(inbox_dir)
    if not d.exists():
        return None

    files = [p for p in d.iterdir() if p.is_file() and p.name.startswith(prefix)]
    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def parse_myf_num(content_id: str) -> Optional[int]:
    base = content_id.split(".")[0]
    m = MYF_RE.match(base)
    if not m:
        return None
    return int(m.group(1))


def extract_myf_ids(available_items: Any) -> list[tuple[int, str]]:
    if isinstance(available_items, dict):
        for key in ("data", "items", "result", "available", "artworks"):
            if key in available_items and isinstance(available_items[key], list):
                available_items = available_items[key]
                break

    if not isinstance(available_items, list):
        return []

    out: list[tuple[int, str]] = []

    for item in available_items:
        if not isinstance(item, dict):
            continue
        cid = item.get("content_id") or item.get("id")
        if not cid:
            continue
        n = parse_myf_num(cid)
        if n is not None:
            out.append((n, cid))

    out.sort(key=lambda x: x[0])
    return out


def extract_content_id(info: Any) -> Optional[str]:
    if not isinstance(info, dict):
        return None

    for key in ("content_id", "id", "contentId", "contentid"):
        cid = info.get(key)
        if cid:
            return str(cid)

    return None


def get_current_info(art: Any) -> dict:
    try:
        current_state = art.get_current() or {}
    except Exception:
        return {}

    if not isinstance(current_state, dict):
        return {}

    current_info = current_state.get("current") if isinstance(current_state.get("current"), dict) else {}
    if not current_info:
        current_info = current_state

    return current_info if isinstance(current_info, dict) else {}


def list_local_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    out: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(".") or name.startswith("._") or name == ".DS_Store":
            continue
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        out.append(p)
    out.sort(key=lambda p: p.name.lower())
    return out


def choose_pick_file(payload: dict, state: dict) -> tuple[Optional[Path], str, int, int]:
    phase = str(payload.get("phase", "")).strip().lower()
    season = str(payload.get("season", "")).strip().lower()
    holiday_raw = str(payload.get("holiday", "none")).strip().lower()
    holiday = HOLIDAY_ALIASES.get(holiday_raw, holiday_raw)

    try:
        rng = int(payload.get("rng", 0))
    except Exception:
        rng = 0

    if holiday and holiday != "none":
        folder = Path("/media/frame_ai/holidays") / holiday
        if holiday in {"christmas", "halloween"}:
            folder = folder / ("evening" if phase in {"evening", "night"} else "day")
    else:
        folder = Path("/media/frame_ai/ambient") / season / phase

    files = list_local_images(folder)
    file_count = len(files)
    if file_count == 0:
        return None, str(folder), 0, -1

    salt = PHASE_SALT.get(phase, 0)
    idx = (rng + salt) % file_count
    if file_count > 1 and str(files[idx]) == str(state.get("last_applied") or ""):
        idx = (idx + 1) % file_count
    return files[idx], str(folder), file_count, idx


def compute_phase_roll(payload: dict) -> tuple[int, int, str]:
    try:
        rng = int(payload.get("rng", 0))
    except Exception:
        rng = 0

    phase_raw = str(payload.get("phase", "")).strip().lower()
    phase = phase_raw if phase_raw in PHASE_SALT else UNKNOWN_PHASE
    salt = PHASE_SALT.get(phase, PHASE_SALT[UNKNOWN_PHASE])
    phase_roll = (rng + salt) % 100
    return rng, phase_roll, phase


def should_pick_samsung_bucket(rng: int, pick_samsung_pct: int) -> tuple[bool, int, int]:
    bucket = abs(int(rng)) % 10
    pct = int(pick_samsung_pct)

    if pct >= 100:
        samsung_buckets = 10
        return True, bucket, samsung_buckets

    if pct <= 0:
        samsung_buckets = 0
        return False, bucket, samsung_buckets

    samsung_buckets = pct // 10
    samsung_buckets = max(0, min(10, samsung_buckets))
    prefer_samsung = bucket < samsung_buckets
    return prefer_samsung, bucket, samsung_buckets


def choose_pick_samsung_id(payload: dict, rng: int, phase: str) -> Optional[str]:
    options = RUNTIME_OPTIONS if isinstance(RUNTIME_OPTIONS, dict) else {}
    samsung_pools = options.get("samsung_pools") if isinstance(options.get("samsung_pools"), dict) else {}
    holidays_pool = samsung_pools.get("holidays") if isinstance(samsung_pools.get("holidays"), dict) else {}
    ambient_pool = samsung_pools.get("ambient") if isinstance(samsung_pools.get("ambient"), dict) else {}

    season = str(payload.get("season", "")).strip().lower()
    holiday_raw = str(payload.get("holiday", "none")).strip().lower()
    holiday = HOLIDAY_ALIASES.get(holiday_raw, holiday_raw)

    pool: list[str] = []

    if holiday and holiday != "none":
        holiday_entry = holidays_pool.get(holiday)
        if holiday in {"christmas", "halloween"} and isinstance(holiday_entry, dict):
            bucket = "evening" if phase in {"evening", "night"} else "day"
            maybe_list = holiday_entry.get(bucket)
            if isinstance(maybe_list, list):
                pool = [str(x).strip() for x in maybe_list if str(x).strip()]
        elif isinstance(holiday_entry, list):
            pool = [str(x).strip() for x in holiday_entry if str(x).strip()]
    else:
        season_entry = ambient_pool.get(season)
        if isinstance(season_entry, dict):
            maybe_list = season_entry.get(phase)
            if isinstance(maybe_list, list):
                pool = [str(x).strip() for x in maybe_list if str(x).strip()]

    if not pool:
        return None

    idx = (rng + PHASE_SALT.get(phase, PHASE_SALT[UNKNOWN_PHASE])) % len(pool)
    return pool[idx]


def parse_restore_request_payload(payload: Any) -> tuple[Optional[dict], Optional[bool], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None, "Invalid restore payload: expected JSON object"

    requested_show = parse_bool(payload.get("show"))

    requested_at_raw = payload.get("requested_at")
    if requested_at_raw:
        try:
            req = datetime.fromisoformat(str(requested_at_raw))
            if req.tzinfo is None:
                req = req.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - req.astimezone(timezone.utc)
            if age > timedelta(hours=8):
                return None, requested_show, "Restore request expired (>8h old)"
        except Exception:
            return None, requested_show, "Malformed restore timestamp"

    normalized = dict(payload)
    kind = str(payload.get("kind", "")).strip().lower()
    if not kind:
        if payload.get("content_id"):
            kind = "content_id"
            normalized["value"] = payload.get("content_id")
        else:
            kind = ""
    normalized["kind"] = kind

    background_mode = str(payload.get("background_mode", "")).strip().lower()
    normalized["background_mode"] = background_mode
    if kind in {"cover_art_reference", "cover_art_reference_background"} or background_mode == "reference-no-mask":
        kind = "cover_art_reference_background"
        normalized["kind"] = kind

    if kind in {"cover_art_outpaint", "cover_art_reference_background"}:
        if requested_show is None:
            requested_show = True
        normalized["requested_at"] = str(payload.get("requested_at", "")).strip()
        normalized["artwork_url"] = str(payload.get("artwork_url", "")).strip()
        normalized["artist"] = str(payload.get("artist", "")).strip()
        normalized["album"] = str(payload.get("album", "")).strip()
        normalized["track"] = str(payload.get("track", "")).strip()
        normalized["music_session_key"] = str(payload.get("music_session_key", "")).strip()
        normalized["key_source"] = str(payload.get("key_source", "")).strip().lower()
        normalized["shazam_key"] = str(payload.get("shazam_key", "")).strip()
        normalized["listening_mode"] = str(payload.get("listening_mode", "")).strip()
        collection_id_raw = payload.get("collection_id")
        if collection_id_raw in (None, ""):
            normalized["collection_id"] = None
        else:
            try:
                normalized["collection_id"] = int(collection_id_raw)
            except Exception:
                normalized["collection_id"] = None

    if kind == "ambient_seed":
        normalized["ambient_dir"] = str(payload.get("ambient_dir", "")).strip()
        normalized["catalog_path"] = str(payload.get("catalog_path", "")).strip()
        normalized["force_reupload"] = bool(payload.get("force_reupload", False))

    return normalized, requested_show, None


def list_ambient_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    out: list[Path] = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        if p.name.startswith(".") or p.name.startswith("._") or p.name == ".DS_Store":
            continue
        if p.suffix.lower() not in AMBIENT_EXTENSIONS:
            continue
        out.append(p)
    out.sort(key=lambda p: p.relative_to(folder).as_posix().lower())
    return out


def load_ambient_catalog(path: Path) -> dict[str, Any]:
    return load_frame_art_catalog(path)


def load_frame_art_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at": "", "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_at": "", "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "updated_at": "", "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    data["version"] = 1
    data["entries"] = entries
    data.setdefault("updated_at", "")
    return data


def persist_ambient_catalog(path: Path, catalog: dict[str, Any]) -> None:
    persist_frame_art_catalog(path, catalog)


def persist_frame_art_catalog(path: Path, catalog: dict[str, Any]) -> None:
    catalog["version"] = 1
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    catalog.setdefault("entries", {})
    atomic_write_json(path, catalog)


def get_catalog_for_local_pick(pick_file: Path) -> tuple[Optional[Path], Optional[str]]:
    try:
        holiday_root = Path("/media/frame_ai/holidays")
        if pick_file.is_relative_to(holiday_root):
            return HOLIDAY_CATALOG_PATH, pick_file.relative_to(holiday_root).as_posix()
    except Exception:
        pass

    try:
        ambient_root = Path("/media/frame_ai/ambient")
        if pick_file.is_relative_to(ambient_root):
            return AMBIENT_CATALOG_PATH, pick_file.relative_to(ambient_root).as_posix()
    except Exception:
        pass

    return None, None


def lookup_catalog_content_id(catalog_path: Path, catalog_key: str) -> Optional[str]:
    catalog = load_frame_art_catalog(catalog_path)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    entry = entries.get(catalog_key) if isinstance(entries.get(catalog_key), dict) else {}
    content_id = str(entry.get("content_id", "")).strip() if entry else ""
    return content_id or None


def update_catalog_content_id(catalog_path: Path, catalog_key: str, content_id: str) -> None:
    catalog = load_frame_art_catalog(catalog_path)
    entries = catalog.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries
    entries[catalog_key] = {
        "content_id": content_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    persist_frame_art_catalog(catalog_path, catalog)


def append_music_error(entry: dict[str, Any]) -> None:
    catalog = load_frame_art_catalog(MUSIC_ERRORS_PATH)
    errors = catalog.get("errors")
    if not isinstance(errors, list):
        errors = []
        catalog["errors"] = errors
    payload = dict(entry)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    errors.append(payload)
    persist_frame_art_catalog(MUSIC_ERRORS_PATH, catalog)


def update_music_association(
    restore_payload: dict[str, Any],
    *,
    cache_key: str,
    catalog_key: str,
    content_id: str,
) -> None:
    catalog = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    entries = catalog.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries

    music_session_key = str(restore_payload.get("music_session_key", "")).strip()
    shazam_key = str(restore_payload.get("shazam_key", "")).strip()
    key_source = str(restore_payload.get("key_source", "")).strip().lower() or "unknown"
    artist = str(restore_payload.get("artist", "")).strip()
    album = str(restore_payload.get("album", "")).strip()
    track = str(restore_payload.get("track", "")).strip()
    collection_id = restore_payload.get("collection_id")

    record = {
        "cache_key": cache_key,
        "catalog_key": catalog_key,
        "content_id": content_id,
        "key_source": key_source,
        "artist": artist,
        "album": album,
        "track": track,
        "collection_id": collection_id if isinstance(collection_id, int) else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if music_session_key:
        entries[f"session::{music_session_key}"] = record
    if shazam_key:
        entries[f"shazam::{shazam_key}"] = record
    if artist or album:
        entries[f"album::{(artist + ' — ' + album).strip()}"] = record
    entries[f"cache::{cache_key}"] = record

    persist_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH, catalog)


def music_catalog_key_for_path(path: Path) -> str:
    try:
        return path.relative_to(COMPRESSED_DIR).as_posix()
    except Exception:
        return path.name


def handle_ambient_seed_restore(tv_ip: str, art: Any, restore_payload: dict, requested_at: str) -> dict[str, Any]:
    ambient_dir_raw = str(restore_payload.get("ambient_dir", "")).strip()
    catalog_path_raw = str(restore_payload.get("catalog_path", "")).strip()
    force_reupload = bool(restore_payload.get("force_reupload", False))

    if not ambient_dir_raw:
        raise ValueError("ambient_seed requires ambient_dir")
    if not catalog_path_raw:
        raise ValueError("ambient_seed requires catalog_path")

    ambient_dir = Path(ambient_dir_raw)
    if not ambient_dir.exists() or not ambient_dir.is_dir():
        raise ValueError(f"Ambient directory missing: {ambient_dir}")

    files = list_ambient_images(ambient_dir)
    if not files:
        raise ValueError(f"Ambient directory has no supported images: {ambient_dir}")

    catalog_path = Path(catalog_path_raw)
    catalog = load_ambient_catalog(catalog_path)
    entries = catalog.setdefault("entries", {})

    uploaded_count = 0
    skipped_count = 0
    failed_count = 0
    failures: list[str] = []
    last_cid: Optional[str] = None

    for image_path in files:
        key = image_path.relative_to(ambient_dir).as_posix()
        entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
        if not entry and "/" in key:
            legacy_key = image_path.name
            if isinstance(entries.get(legacy_key), dict):
                entry = entries.get(legacy_key)
        cached_id = str(entry.get("content_id", "")).strip() if entry else ""

        if cached_id and not force_reupload:
            skipped_count += 1
            last_cid = cached_id
            continue

        try:
            art, content_id = upload_local_file_with_reconnect(tv_ip, art, image_path)
            if not content_id:
                raise ValueError("content_id missing after upload")
            uploaded_count += 1
            last_cid = content_id
            entries[key] = {
                "content_id": content_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            persist_ambient_catalog(catalog_path, catalog)
        except Exception as exc:
            failed_count += 1
            failures.append(f"{key}: {repr(exc)}")

    ok = failed_count == 0 and (uploaded_count > 0 or skipped_count > 0)
    return {
        "ok": ok,
        "mode": "restore",
        "kind": "ambient_seed",
        "requested_at": requested_at,
        "ambient_dir": str(ambient_dir),
        "catalog_path": str(catalog_path),
        "force_reupload": force_reupload,
        "total_files": len(files),
        "uploaded_count": uploaded_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "selected_content_id": last_cid,
        "error": None if ok else "; ".join(failures) if failures else "ambient_seed completed with no successful uploads",
    }


def _queue_sort_key(path: Path) -> tuple[float, str]:
    try:
        return path.stat().st_mtime, path.name
    except Exception:
        return 0.0, path.name


def list_queued_requests() -> list[Path]:
    if not RESTORE_QUEUE_DIR.exists():
        return []
    files = [p for p in RESTORE_QUEUE_DIR.iterdir() if p.is_file() and p.suffix == ".json"]
    return sorted(files, key=_queue_sort_key)


def enqueue_restore_inbox_if_present() -> Optional[Path]:
    inbox = Path(RESTORE_REQUEST_PATH)
    if not inbox.exists():
        return None

    RESTORE_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    unique = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json"
    queued = RESTORE_QUEUE_DIR / unique
    os.replace(inbox, queued)
    return queued


def dequeue_next_restore_work_item() -> Optional[Path]:
    queued = list_queued_requests()
    if not queued:
        return None
    return queued[0]


@contextmanager
def worker_lock() -> Any:
    WORKER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WORKER_LOCK_PATH, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            got_lock = True
        except BlockingIOError:
            got_lock = False
        try:
            yield got_lock
        finally:
            if got_lock:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_restore_work_item(path: Path) -> tuple[Optional[dict], Optional[bool], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, "Malformed restore JSON"

    return parse_restore_request_payload(payload)


def prepare_for_frame(img_bytes: bytes) -> tuple[bytes, str]:
    im = Image.open(BytesIO(img_bytes))

    if im.mode not in {"RGB", "RGBA", "L"}:
        im = im.convert("RGB")

    w, h = im.size
    target_ratio = 16 / 9
    current_ratio = w / h if h else target_ratio

    if current_ratio > target_ratio:
        target_w = int(round(h * target_ratio))
        left = (w - target_w) // 2
        im = im.crop((left, 0, left + target_w, h))
    elif current_ratio < target_ratio:
        target_h = int(round(w / target_ratio))
        top = (h - target_h) // 2
        im = im.crop((0, top, w, top + target_h))

    im = im.resize((3840, 2160), Image.Resampling.LANCZOS)

    # Prefer JPEG for speed/size tradeoffs, but do not enforce any add-on-side
    # upload byte cap. The TV should be the source of truth for capacity limits.
    if im.mode != "RGB":
        im = im.convert("RGB")

    out = BytesIO()
    im.save(out, format="JPEG", quality=92, optimize=True, progressive=True)
    jpeg_bytes = out.getvalue()

    # Fallback to PNG only if JPEG encoding fails to preserve compatibility
    # with callers expecting one of the two supported file types.
    if jpeg_bytes:
        return jpeg_bytes, "JPEG"

    out = BytesIO()
    im.save(out, format="PNG", optimize=True, compress_level=6)
    return out.getvalue(), "PNG"


def resolve_local_path(value: str) -> Optional[Path]:
    p = Path(value).resolve()
    base = Path("/media/frame_ai").resolve()
    if not str(p).startswith(str(base) + "/") and p != base:
        return None
    if not p.exists() or not p.is_file():
        return None
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    if p.name.startswith(".") or p.name.startswith("._") or p.name == ".DS_Store":
        return None
    return p


def pick_cover_fallback(cache_key: str) -> Optional[Path]:
    candidates = [
        FALLBACK_DIR / f"{cache_key}.png",
        FALLBACK_DIR / f"{cache_key}.jpg",
        FALLBACK_DIR / "default.png",
        FALLBACK_DIR / "default.jpg",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def upload_local_file(art: Any, file_path: Path) -> Optional[str]:
    raw_bytes = file_path.read_bytes()
    processed_bytes, file_type = prepare_for_frame(raw_bytes)
    art.upload(processed_bytes, file_type=file_type, matte="none")
    available = art.available()
    myf = extract_myf_ids(available)
    return myf[-1][1] if myf else None


def is_broken_pipe_error(exc: Exception) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    message = repr(exc).lower()
    return "broken pipe" in message


def upload_local_file_with_reconnect(tv_ip: str, art: Any, file_path: Path) -> tuple[Any, Optional[str]]:
    try:
        return art, upload_local_file(art, file_path)
    except Exception as exc:
        if not is_broken_pipe_error(exc):
            raise

        log_event("upload_retry", reason=repr(exc), file=str(file_path), action="reconnect_art_socket")
        retry_tv = SamsungTVWS(tv_ip)
        retry_art = retry_tv.art()
        if not retry_art.supported():
            raise
        return retry_art, upload_local_file(retry_art, file_path)


def cleanup_local_uploads(art: Any, state: dict, keep_count_local: int) -> tuple[list[str], Optional[str]]:
    return cleanup_frame_uploads(art, state, "local_uploaded_ids", keep_count_local)


def cleanup_frame_uploads(art: Any, state: dict, state_key: str, keep_count_local: int) -> tuple[list[str], Optional[str]]:
    tracked = sanitize_local_uploaded_ids(state.get(state_key))
    state[state_key] = tracked
    return [], None


def run_pending_keep_count_cleanup(art: Any, state: dict, keep_count: int) -> list[str]:
    state["pending_keep_count_cleanup"] = False
    save_state(state)
    return []


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    global RUNTIME_OPTIONS
    opts = load_options()
    RUNTIME_OPTIONS = opts if isinstance(opts, dict) else {}
    state = load_state()

    tv_ip = str(opts.get("tv_ip", "")).strip()
    keep_count = int(opts.get("keep_count", 20))
    keep_count_local = int(opts.get("keep_count_local", 30))
    try:
        pick_samsung_pct = int(opts.get("pick_samsung_pct", 60))
    except Exception:
        pick_samsung_pct = 60
    pick_samsung_pct = max(0, min(100, pick_samsung_pct))
    inbox_dir = str(opts.get("inbox_dir", "/media/frame_ai/inbox"))
    prefix = str(opts.get("filename_prefix", "ai_"))
    select_after = bool(opts.get("select_after_upload", True))
    openai_api_key = str(opts.get("openai_api_key", "")).strip()
    openai_model = str(opts.get("openai_model", "gpt-image-1")).strip() or "gpt-image-1"

    if not tv_ip:
        write_status({"ok": False, "error": "Missing tv_ip"})
        return

    tv = SamsungTVWS(tv_ip)
    art = tv.art()

    if not art.supported():
        write_status(
            {
                "ok": False,
                "error": "Art mode not supported / unreachable",
                "tv_ip": tv_ip,
            }
        )
        return

    enqueue_restore_inbox_if_present()

    with worker_lock() as has_lock:
        if not has_lock:
            write_status(
                {
                    "ok": True,
                    "mode": "queue",
                    "kind": "queue_busy",
                    "requested_at": "",
                    "error": "worker_already_processing",
                }
            )
            return

        handled_restore_work = False
        while True:
            enqueue_restore_inbox_if_present()
            work_item = dequeue_next_restore_work_item()
            if work_item is None:
                break
            handled_restore_work = True

            requested_at = ""
            payload_kind = None
            try:
                restore_payload, requested_show, parse_error = load_restore_work_item(work_item)
                if restore_payload:
                    payload_kind = str(restore_payload.get("kind", "")).strip().lower() or None
                    requested_at = str(restore_payload.get("requested_at", "")).strip()
                if parse_error:
                    raise ValueError(parse_error)
                if not restore_payload:
                    raise ValueError("Invalid restore payload")

                current_info = get_current_info(art)
                is_art_mode = bool(current_info) or extract_content_id(current_info) is not None
                show_flag = requested_show if requested_show is not None else is_art_mode

                kind = str(restore_payload.get("kind", "")).strip().lower()
                request_value = str(restore_payload.get("value", "")).strip()
                resolved_folder = None
                file_count = 0
                chosen_index = -1
                selected_name = None
                pick_source = None
                encoded_type = None
                encoded_bytes = None
                rng = None
                phase_roll = None
                bucket = None
                samsung_buckets = None

                if kind == "content_id":
                    target_cid = request_value
                    if not target_cid:
                        raise ValueError(
                            f"Invalid restore request: missing content_id value (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                        )
                    selected_name = target_cid
                elif kind == "local_file":
                    local_path = resolve_local_path(request_value)
                    if local_path is None:
                        raise ValueError(
                            f"Invalid restore request: local_file must be an existing jpg/png under /media/frame_ai (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                        )
                    art, target_cid = upload_local_file_with_reconnect(tv_ip, art, local_path)
                    if not target_cid:
                        raise ValueError(
                            f"Upload completed but content_id was not discovered (kind={kind}, value={request_value!r}, resolved_folder={local_path.parent})"
                        )
                    append_local_uploaded_id(state, target_cid)
                    state["last_applied"] = str(local_path)
                    resolved_folder = str(local_path.parent)
                    file_count = 1
                    chosen_index = 0
                    selected_name = local_path.name
                elif kind == "pick":
                    rng, phase_roll, phase = compute_phase_roll(restore_payload)
                    prefer_samsung, bucket, samsung_buckets = should_pick_samsung_bucket(rng, pick_samsung_pct)

                    if prefer_samsung:
                        target_cid = choose_pick_samsung_id(restore_payload, rng, phase)
                        if target_cid:
                            pick_source = "samsung"
                            resolved_folder = "samsung_pool"
                            selected_name = target_cid
                        else:
                            pick_source = "local"
                    else:
                        pick_source = "local"

                    if pick_source == "local":
                        pick_file, resolved_folder, file_count, chosen_index = choose_pick_file(restore_payload, state)
                        if not pick_file:
                            raise ValueError(
                                f"No candidate files found for pick request (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                            )

                        selected_name = pick_file.name
                        catalog_path, catalog_key = get_catalog_for_local_pick(pick_file)
                        catalog_hit = False

                        if catalog_path and catalog_key:
                            cached_content_id = lookup_catalog_content_id(catalog_path, catalog_key)
                            if cached_content_id:
                                target_cid = cached_content_id
                                catalog_hit = True
                                log_event(
                                    "catalog_pick_hit",
                                    file=str(pick_file),
                                    catalog_path=str(catalog_path),
                                    catalog_key=catalog_key,
                                    content_id=target_cid,
                                )

                        if not catalog_hit:
                            art, target_cid = upload_local_file_with_reconnect(tv_ip, art, pick_file)
                            if not target_cid:
                                raise ValueError(
                                    f"Upload completed but content_id was not discovered (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                                )
                            append_local_uploaded_id(state, target_cid)
                            if catalog_path and catalog_key:
                                update_catalog_content_id(catalog_path, catalog_key, target_cid)
                                log_event(
                                    "catalog_pick_update",
                                    file=str(pick_file),
                                    catalog_path=str(catalog_path),
                                    catalog_key=catalog_key,
                                    content_id=target_cid,
                                )

                        state["last_applied"] = str(pick_file)
                    else:
                        state["last_applied"] = f"samsung:{target_cid}"
                elif kind == "ambient_seed":
                    ambient_status = handle_ambient_seed_restore(tv_ip, art, restore_payload, requested_at)
                    write_status({**ambient_status, "tv_ip": tv_ip, "addon_version": ADDON_VERSION})
                    continue
                elif kind in {"cover_art_reference_background", "cover_art_outpaint"}:
                    ensure_dirs()
                    source_url = str(restore_payload.get("artwork_url", "")).strip()
                    artist = str(restore_payload.get("artist", "")).strip()
                    album = str(restore_payload.get("album", "")).strip()
                    collection_id_raw = restore_payload.get("collection_id")
                    collection_id: Optional[int] = None
                    if collection_id_raw not in (None, ""):
                        try:
                            collection_id = int(collection_id_raw)
                        except Exception:
                            collection_id = None

                    cache_key = normalize_key(collection_id, artist, album)
                    stem_key = str(collection_id) if isinstance(collection_id, int) else cache_key
                    src_path = SOURCE_DIR / f"{stem_key}.jpg"
                    background_path = BACKGROUND_DIR / f"{stem_key}__3840x2160__background.png"
                    wide_png_path = WIDESCREEN_DIR / f"{stem_key}__3840x2160.png"
                    wide_jpg_path = WIDESCREEN_DIR / f"{stem_key}__3840x2160.jpg"
                    compressed_jpg_path = COMPRESSED_DIR / f"{stem_key}__3840x2160.jpg"

                    # Backward-compat candidates from older cover_art layout.
                    legacy_png_path = WIDESCREEN_DIR / f"{cache_key}.png"
                    legacy_jpg_path = WIDESCREEN_DIR / f"{cache_key}.jpg"
                    wide_path = compressed_jpg_path
                    if not wide_path.exists():
                        for candidate in (wide_png_path, wide_jpg_path, legacy_png_path, legacy_jpg_path):
                            if candidate.exists():
                                wide_path = candidate
                                break

                    resolved_folder = str(WIDESCREEN_DIR)
                    selected_name = wide_path.name
                    file_count = 1
                    chosen_index = 0

                    if wide_path.exists():
                        catalog_key = music_catalog_key_for_path(wide_path)
                        cached_content_id = lookup_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key)
                        if cached_content_id:
                            target_cid = cached_content_id
                            encoded_type = guess_file_type(wide_path)
                            encoded_bytes = wide_path.stat().st_size
                            pick_source = "music_cache_catalog_hit"
                        else:
                            encoded_type = guess_file_type(wide_path)
                            encoded_bytes = wide_path.stat().st_size
                            art, target_cid = upload_local_file_with_reconnect(tv_ip, art, wide_path)
                            if not target_cid:
                                raise ValueError("Cached music upload succeeded but content_id was not found")
                            append_uploaded_id(state, "cover_uploaded_ids", target_cid)
                            update_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key, target_cid)
                            update_music_association(
                                restore_payload,
                                cache_key=cache_key,
                                catalog_key=catalog_key,
                                content_id=target_cid,
                            )
                            pick_source = "music_cache_uploaded"
                    else:
                        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
                        cover_error: Optional[Exception] = None
                        fallback_path: Optional[Path] = None
                        try:
                            if not source_url:
                                if collection_id is not None:
                                    lookup = itunes_lookup(collection_id, timeout_s=10)
                                    results = lookup.get("results") if isinstance(lookup, dict) else []
                                    album_info = {}
                                    if isinstance(results, list):
                                        for item in results:
                                            if isinstance(item, dict) and (item.get("artworkUrl100") or item.get("artworkUrl60")):
                                                album_info = item
                                                break
                                    source_url = resolve_artwork_url(album_info)
                                elif artist and album:
                                    album_info = itunes_search(artist, album, timeout_s=10)
                                    source_url = resolve_artwork_url(album_info)
                                else:
                                    raise ValueError("unsupported metadata; provide artwork_url, collection_id, or artist+album")

                            if not source_url:
                                raise ValueError("Unable to resolve artwork URL from request metadata")

                            if not src_path.exists():
                                download_artwork(source_url, str(src_path), timeout_s=15)

                            use_legacy_masked = bool(RUNTIME_OPTIONS.get("legacy_masked_outpaint", False))
                            if use_legacy_masked:
                                log_event("legacy_masked_outpaint_requested", enabled=True, note="using reference_no_mask pipeline")

                            try:
                                final_png, background_png, request_id, model_used = generate_reference_frame_from_album(
                                    source_album_path=src_path,
                                    openai_api_key=openai_api_key,
                                    openai_model=openai_model,
                                    timeout_s=90,
                                    album_shadow=True,
                                )
                                generation_mode = "openai_reference"
                            except Exception as gen_exc:
                                gen_msg = repr(gen_exc).lower()
                                if "moderation_blocked" in gen_msg or "safety system" in gen_msg:
                                    final_png, background_png = generate_local_fallback_frame_from_album(
                                        source_album_path=src_path,
                                        album_shadow=True,
                                    )
                                    request_id = None
                                    model_used = "local-fallback"
                                    generation_mode = "local_fallback_blocked"
                                else:
                                    raise

                            wide_png_path.parent.mkdir(parents=True, exist_ok=True)
                            background_path.parent.mkdir(parents=True, exist_ok=True)
                            compressed_jpg_path.parent.mkdir(parents=True, exist_ok=True)
                            wide_png_path.write_bytes(final_png)
                            background_path.write_bytes(background_png)
                            compressed_ok, compressed_size = compress_png_path_to_jpeg_max_bytes(
                                wide_png_path,
                                compressed_jpg_path,
                                max_bytes=JPEG_MAX_BYTES,
                            )
                            wide_path = compressed_jpg_path if compressed_jpg_path.exists() else wide_png_path
                            encoded_type = guess_file_type(wide_path)
                            encoded_bytes = wide_path.stat().st_size
                            art, target_cid = upload_local_file_with_reconnect(tv_ip, art, wide_path)
                            if not target_cid:
                                raise ValueError("Generated music upload succeeded but content_id was not found")
                            append_uploaded_id(state, "cover_uploaded_ids", target_cid)
                            catalog_key = music_catalog_key_for_path(wide_path)
                            update_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key, target_cid)
                            update_music_association(
                                restore_payload,
                                cache_key=cache_key,
                                catalog_key=catalog_key,
                                content_id=target_cid,
                            )
                            pick_source = generation_mode
                            log_event(
                                "music_generated",
                                cache_key=cache_key,
                                mode=generation_mode,
                                compressed_within_limit=compressed_ok,
                                compressed_bytes=compressed_size,
                                path=str(wide_path),
                                background_path=str(background_path),
                            )
                        except Exception as e:
                            cover_error = e
                            append_music_error(
                                {
                                    "cache_key": cache_key,
                                    "collection_id": collection_id,
                                    "artist": artist,
                                    "album": album,
                                    "track": track,
                                    "music_session_key": restore_payload.get("music_session_key", ""),
                                    "key_source": restore_payload.get("key_source", ""),
                                    "error": repr(e),
                                }
                            )
                            fallback_path = pick_cover_fallback(cache_key)
                            if fallback_path is None:
                                raise
                            encoded_type = guess_file_type(fallback_path)
                            encoded_bytes = fallback_path.stat().st_size
                            art, target_cid = upload_local_file_with_reconnect(tv_ip, art, fallback_path)
                            if not target_cid:
                                raise ValueError("Fallback cover upload succeeded but content_id was not found")
                            append_uploaded_id(state, "cover_uploaded_ids", target_cid)
                            pick_source = "music_fallback_file"
                            log_event("music_fallback_file", cache_key=cache_key, error=repr(cover_error), fallback=str(fallback_path))
                else:
                    raise ValueError(f"Unsupported restore kind: {kind!r}")

                verified = False
                verification_skipped = False

                if show_flag and target_cid:
                    current_id = extract_content_id(current_info)
                    if current_id == target_cid:
                        verified = True
                    else:
                        pick_local_retry_file: Optional[Path] = None
                        pick_local_retry_catalog_path: Optional[Path] = None
                        pick_local_retry_catalog_key: Optional[str] = None

                        if kind == "pick" and pick_source == "local":
                            pick_file_candidate, _, _, _ = choose_pick_file(restore_payload, state)
                            if pick_file_candidate and str(pick_file_candidate.name) == str(selected_name):
                                pick_local_retry_file = pick_file_candidate
                                pick_local_retry_catalog_path, pick_local_retry_catalog_key = get_catalog_for_local_pick(
                                    pick_file_candidate
                                )

                        select_attempt_error: Optional[Exception] = None
                        try:
                            art.select_image(target_cid, show=True)
                            for _ in range(3):
                                time.sleep(1.0)
                                current_info = get_current_info(art)
                                current_id = extract_content_id(current_info)
                                if current_id == target_cid:
                                    verified = True
                                    break
                                art.select_image(target_cid, show=True)
                        except Exception as select_exc:
                            select_attempt_error = select_exc

                        if not verified and pick_local_retry_file is not None:
                            log_event(
                                "catalog_pick_fallback_upload",
                                file=str(pick_local_retry_file),
                                selected_content_id=target_cid,
                                reason=repr(select_attempt_error) if select_attempt_error else "select_unverified",
                            )
                            art, replacement_cid = upload_local_file_with_reconnect(tv_ip, art, pick_local_retry_file)
                            if not replacement_cid:
                                raise ValueError(
                                    f"Upload fallback completed but content_id was not discovered (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                                )
                            append_local_uploaded_id(state, replacement_cid)
                            target_cid = replacement_cid
                            if pick_local_retry_catalog_path and pick_local_retry_catalog_key:
                                update_catalog_content_id(
                                    pick_local_retry_catalog_path,
                                    pick_local_retry_catalog_key,
                                    replacement_cid,
                                )
                            art.select_image(target_cid, show=True)
                            for _ in range(3):
                                time.sleep(1.0)
                                current_info = get_current_info(art)
                                current_id = extract_content_id(current_info)
                                if current_id == target_cid:
                                    verified = True
                                    break
                                art.select_image(target_cid, show=True)
                else:
                    verification_skipped = True

                deleted_local: list[str] = []
                cleanup_error: Optional[str] = None
                if kind == "pick" and pick_source == "local":
                    deleted_local, cleanup_error = cleanup_local_uploads(art, state, keep_count_local)
                elif kind == "local_file":
                    deleted_local, cleanup_error = cleanup_local_uploads(art, state, keep_count_local)
                elif kind in {"cover_art_reference_background", "cover_art_outpaint"}:
                    deleted_local, cleanup_error = cleanup_frame_uploads(art, state, "cover_uploaded_ids", keep_count_local)

                deleted_keep_count: list[str] = []
                pending_cleanup_error: Optional[str] = None
                try:
                    deleted_keep_count = run_pending_keep_count_cleanup(art, state, keep_count)
                except Exception as e:
                    pending_cleanup_error = repr(e)

                save_state(state)

                log_event(
                    "restore_request",
                    kind=kind,
                    resolved_folder=resolved_folder,
                    file_count=file_count,
                    chosen_index=chosen_index,
                    chosen=selected_name or target_cid,
                    pick_source=pick_source,
                    rng=rng,
                    phase_roll=phase_roll,
                    bucket=bucket,
                    samsung_buckets=samsung_buckets,
                    pick_samsung_pct=pick_samsung_pct,
                    cleanup_deletions=len(deleted_local),
                    cleanup_error=cleanup_error,
                    keep_count_cleanup_deletions=len(deleted_keep_count),
                    keep_count_cleanup_error=pending_cleanup_error,
                )

                write_status(
                    {
                        "ok": True if not show_flag else bool(verified),
                        "mode": "restore",
                        "tv_ip": tv_ip,
                        "kind": kind,
                        "value": request_value,
                        "requested_at": requested_at,
                        "resolved_folder": resolved_folder,
                        "file_count": file_count,
                        "chosen_index": chosen_index,
                        "requested_show": requested_show,
                        "art_mode": is_art_mode,
                        "pick_source": pick_source,
                        "rng": rng,
                        "phase_roll": phase_roll,
                        "bucket": bucket,
                        "samsung_buckets": samsung_buckets,
                        "pick_samsung_pct": pick_samsung_pct,
                        "addon_version": ADDON_VERSION,
                        "selected_content_id": target_cid,
                        "chosen": selected_name or target_cid,
                        "deleted_local": deleted_local,
                        "cleanup_error": cleanup_error,
                        "keep_count_deleted": deleted_keep_count,
                        "keep_count_cleanup_error": pending_cleanup_error,
                        "verified": verified,
                        "verification_skipped": verification_skipped,
                        "cache_key": locals().get("cache_key"),
                        "source_path": str(locals().get("src_path")) if "src_path" in locals() else None,
                        "widescreen_path": str(locals().get("wide_path")) if "wide_path" in locals() else None,
                        "fallback_path": str(locals().get("fallback_path")) if "fallback_path" in locals() and locals().get("fallback_path") else None,
                        "encoded_type": encoded_type if kind in {"cover_art_outpaint", "cover_art_reference_background"} else None,
                        "encoded_bytes": encoded_bytes if kind in {"cover_art_outpaint", "cover_art_reference_background"} else None,
                        "prompt_variant": "reference_background_nomask" if kind in {"cover_art_outpaint", "cover_art_reference_background"} else None,
                        "pipeline": "reference_no_mask" if kind in {"cover_art_outpaint", "cover_art_reference_background"} else None,
                        "mask_mode": "none" if kind in {"cover_art_outpaint", "cover_art_reference_background"} else None,
                        "openai_request_id": locals().get("request_id"),
                        "openai_model_used": locals().get("model_used"),
                    }
                )
            except Exception as e:
                write_status(
                    {
                        "ok": False,
                        "mode": "restore",
                        "tv_ip": tv_ip,
                        "kind": payload_kind,
                        "requested_at": requested_at,
                        "error": repr(e),
                    }
                )
            finally:
                try:
                    work_item.unlink()
                except Exception:
                    pass

        if handled_restore_work:
            return

    img_path = newest_candidate(inbox_dir, prefix)
    if img_path is None:
        write_status(
            {
                "ok": False,
                "error": "No matching image found",
                "inbox_dir": inbox_dir,
                "prefix": prefix,
            }
        )
        return

    raw_bytes = img_path.read_bytes()
    original_type = guess_file_type(img_path)
    processed_bytes, file_type = prepare_for_frame(raw_bytes)

    art.upload(processed_bytes, file_type=file_type, matte="none")

    available = art.available()
    myf = extract_myf_ids(available)
    newest = myf[-1][1] if myf else None

    deleted: list[str] = []
    cleanup_scheduled = False

    if select_after and newest:
        art.select_image(newest, show=True)

    write_status(
        {
            "ok": True,
            "mode": "upload",
            "tv_ip": tv_ip,
            "uploaded_file": str(img_path),
            "original_file_type": original_type,
            "uploaded_file_type": file_type,
            "selected_content_id": newest,
            "keep_count": keep_count,
            "myf_count": len(myf),
            "deleted": deleted,
            "keep_count_cleanup_scheduled": cleanup_scheduled,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_status({"ok": False, "error": repr(e)})
        raise
