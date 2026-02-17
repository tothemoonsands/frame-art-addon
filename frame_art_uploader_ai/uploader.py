import json
import os
import re
import shutil
import time
import uuid
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from samsungtvws import SamsungTVWS
try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except Exception:
    rapidfuzz_fuzz = None

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
MUSIC_TRIAGE_PATH = Path("/share/frame_art_music_triage.json")
MUSIC_OVERRIDES_PATH = Path("/share/frame_art_music_overrides.json")
MUSIC_FEEDBACK_QUEUE_DIR = Path("/share/frame_art_music_feedback_queue")
MUSIC_INDEX_PATHS = [
    Path("/media/frame_ai/music/index.json"),
    Path("/root/media/frame_ai/music/index.json"),
]
MUSIC_MANIFEST_PATHS = [
    Path("/media/frame_ai/music/manifest.json"),
    Path("/root/media/frame_ai/music/manifest.json"),
    Path("/media/frame_ai/music/manifest.jsonl"),
    Path("/root/media/frame_ai/music/manifest.jsonl"),
]

MYF_RE = re.compile(r"^MY[_-]F(\d+)", re.IGNORECASE)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AMBIENT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEED_KINDS = {"ambient_seed", "holiday_seed", "music_seed"}
DEFAULT_SEED_DELETE_LIMIT = 25
PHASE_SALT = {
    "pre_dawn": 101,
    "midday": 303,
    "evening": 505,
    "night": 707,
}
UNKNOWN_PHASE = "night"
MUSIC_RESTORE_KINDS = {"cover_art_reference_background", "cover_art_outpaint"}
# Deprecated: session/shazam aliases no longer expire.
MUSIC_ASSOCIATION_SESSION_TTL_DAYS = 0

RUNTIME_OPTIONS: dict[str, Any] = {}
ADDON_VERSION = "1.6"
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


def parse_shazam_album_match_key(raw_value: Any) -> tuple[str, str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return "", ""
    body = raw
    if raw.lower().startswith("album:"):
        body = raw.split(":", 1)[1]
    if "|" not in body:
        return "", ""
    artist_raw, album_raw = body.split("|", 1)
    artist = str(artist_raw).strip().strip("\"'").strip()
    album = str(album_raw).strip().strip("\"'").strip()
    if not (artist and album):
        return "", ""
    return artist, album


def music_metadata_looks_low_quality(artist: str, album: str) -> bool:
    artist_norm = normalize_music_text(artist)
    album_norm = normalize_music_text(album)
    if not artist_norm or not album_norm:
        return True
    if artist_norm in {"artist", "unknown"}:
        return True
    if album_norm in {"album", "unknown", "music"}:
        return True
    return False


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
        artist = str(payload.get("artist", "")).strip()
        album = str(payload.get("album", "")).strip()
        track = str(payload.get("track", "")).strip()
        music_session_key = str(payload.get("music_session_key", "")).strip()
        shazam_key = str(payload.get("shazam_key", "")).strip()
        key_source_raw = str(payload.get("key_source", "")).strip().lower()

        shazam_like = any(
            str(v).strip().lower().startswith("album:")
            for v in (shazam_key, music_session_key)
            if str(v).strip()
        )
        inferred_shazam = key_source_raw == "shazam" or (
            key_source_raw in {"true", "1", "yes", "on"} and shazam_like
        )
        key_source = "shazam" if inferred_shazam else key_source_raw
        normalized["music_session_key"] = music_session_key
        normalized["key_source"] = key_source
        # Prevent stale Shazam keys from being reused when the current key source is not Shazam.
        normalized["shazam_key"] = shazam_key if inferred_shazam else ""

        parsed_artist = ""
        parsed_album = ""
        for raw_key in (shazam_key, music_session_key):
            parsed_artist, parsed_album = parse_shazam_album_match_key(raw_key)
            if parsed_artist and parsed_album:
                break

        if parsed_artist and parsed_album:
            if music_metadata_looks_low_quality(artist, album):
                artist = parsed_artist
                album = parsed_album
            elif key_source == "shazam" and track and normalize_music_text(track) == normalize_music_text(parsed_artist):
                artist = parsed_artist
                album = parsed_album

        normalized["artist"] = artist
        normalized["album"] = album
        normalized["track"] = track
        normalized["listening_mode"] = str(payload.get("listening_mode", "")).strip()
        normalized["source_preference"] = str(payload.get("source_preference", "")).strip().lower()
        collection_id_raw = payload.get("collection_id")
        if collection_id_raw in (None, ""):
            normalized["collection_id"] = None
        else:
            try:
                normalized["collection_id"] = int(collection_id_raw)
            except Exception:
                normalized["collection_id"] = None
        normalized["force_regen"] = bool(payload.get("force_regen", False))

    if kind == "music_feedback":
        if requested_show is None:
            requested_show = True
        normalized["requested_at"] = str(payload.get("requested_at", "")).strip()
        normalized["action"] = str(payload.get("action", "queue_only")).strip().lower() or "queue_only"
        normalized["issue_type"] = str(payload.get("issue_type", "")).strip().lower()
        normalized["artist"] = str(payload.get("artist", "")).strip()
        normalized["album"] = str(payload.get("album", "")).strip()
        normalized["track"] = str(payload.get("track", "")).strip()
        normalized["music_session_key"] = str(payload.get("music_session_key", "")).strip()
        normalized["listening_mode"] = str(payload.get("listening_mode", "")).strip()
        normalized["key_source"] = str(payload.get("key_source", "")).strip().lower()
        normalized["shazam_key"] = str(payload.get("shazam_key", "")).strip()
        normalized["cache_key"] = str(payload.get("cache_key", "")).strip()
        normalized["current_content_id"] = str(payload.get("current_content_id", "")).strip()
        normalized["candidate_catalog_key"] = str(payload.get("candidate_catalog_key", "")).strip()
        normalized["notes"] = str(payload.get("notes", "")).strip()
        collection_id_raw = payload.get("collection_id")
        if collection_id_raw in (None, ""):
            normalized["collection_id"] = None
        else:
            try:
                normalized["collection_id"] = int(collection_id_raw)
            except Exception:
                normalized["collection_id"] = None

    if kind in SEED_KINDS:
        normalized["force_reupload"] = bool(payload.get("force_reupload", False))
        normalized["apply_deletions"] = bool(payload.get("apply_deletions", True))
        normalized["auto_queue_missing"] = bool(payload.get("auto_queue_missing", True))
        try:
            normalized["delete_limit"] = max(1, int(payload.get("delete_limit", DEFAULT_SEED_DELETE_LIMIT)))
        except Exception:
            normalized["delete_limit"] = DEFAULT_SEED_DELETE_LIMIT

    if kind == "ambient_seed":
        normalized["ambient_dir"] = str(payload.get("ambient_dir", "")).strip()
        normalized["catalog_path"] = str(payload.get("catalog_path", "")).strip()
    elif kind == "holiday_seed":
        normalized["holiday_dir"] = str(payload.get("holiday_dir", "")).strip()
        normalized["catalog_path"] = str(payload.get("catalog_path", "")).strip()
    elif kind == "music_seed":
        normalized["music_dir"] = str(payload.get("music_dir", "")).strip()
        normalized["catalog_path"] = str(payload.get("catalog_path", "")).strip()

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


def load_frame_art_catalog_with_validity(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {"version": 1, "updated_at": "", "entries": {}}, True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_at": "", "entries": {}}, False
    if not isinstance(data, dict):
        return {"version": 1, "updated_at": "", "entries": {}}, False
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    data["version"] = 1
    data["entries"] = entries
    data.setdefault("updated_at", "")
    return data, True


def persist_ambient_catalog(path: Path, catalog: dict[str, Any]) -> None:
    persist_frame_art_catalog(path, catalog)


def persist_frame_art_catalog(path: Path, catalog: dict[str, Any]) -> None:
    catalog["version"] = 1
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    catalog.setdefault("entries", {})
    atomic_write_json(path, catalog)


def seed_catalog_entry(raw_entry: Any) -> dict[str, Any]:
    entry = dict(raw_entry) if isinstance(raw_entry, dict) else {}
    state = str(entry.get("state", "active")).strip().lower()
    if state not in {"active", "pending_delete", "deleted"}:
        state = "active"
    entry["state"] = state
    entry.setdefault("delete_requested_at", "")
    entry.setdefault("delete_reason", "")
    entry.setdefault("deleted_at", "")
    entry.setdefault("updated_at", "")
    entry.setdefault("content_id", "")
    return entry


def mark_catalog_entry_pending_delete(entries: dict[str, Any], key: str, *, reason: str) -> bool:
    existing = seed_catalog_entry(entries.get(key))
    if existing.get("state") == "deleted":
        return False
    if existing.get("state") == "pending_delete":
        return False
    existing["state"] = "pending_delete"
    existing["delete_requested_at"] = datetime.now(timezone.utc).isoformat()
    existing["delete_reason"] = reason
    entries[key] = existing
    return True


def mark_catalog_entry_deleted(entries: dict[str, Any], key: str) -> None:
    existing = seed_catalog_entry(entries.get(key))
    existing["state"] = "deleted"
    existing["deleted_at"] = datetime.now(timezone.utc).isoformat()
    existing["content_id"] = ""
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    entries[key] = existing


def set_catalog_entry_active(entries: dict[str, Any], key: str, content_id: str) -> None:
    entries[key] = {
        "content_id": content_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": "active",
        "delete_requested_at": "",
        "delete_reason": "",
        "deleted_at": "",
    }


def list_seed_images(folder: Path) -> list[Path]:
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


def available_content_ids(art: Any) -> set[str]:
    try:
        available = art.available()
    except Exception:
        return set()
    myf = extract_myf_ids(available)
    return {cid for _, cid in myf}


def delete_art_content_id(art: Any, content_id: str) -> dict[str, Any]:
    cid = str(content_id or "").strip()
    if not cid:
        return {"ok": True, "verified": True, "error": None}

    method_errors: list[str] = []
    deleted_call = False

    # Match samsungtv_artmode's concrete delete path:
    # delete(content_id) -> delete_list([content_id]) -> request "delete_image_list".
    delete_list = getattr(art, "delete_list", None)
    if callable(delete_list):
        try:
            delete_list([cid])
            deleted_call = True
        except Exception as exc:
            method_errors.append(f"delete_list:{repr(exc)}")

    if not deleted_call:
        delete = getattr(art, "delete", None)
        if callable(delete):
            try:
                delete(cid)
                deleted_call = True
            except Exception as exc:
                method_errors.append(f"delete:{repr(exc)}")

    if not deleted_call:
        delete_image = getattr(art, "delete_image", None)
        if callable(delete_image):
            try:
                delete_image(cid)
                deleted_call = True
            except Exception as exc:
                method_errors.append(f"delete_image:{repr(exc)}")

    remaining_ids = available_content_ids(art)
    verified = cid not in remaining_ids
    if verified:
        return {"ok": True, "verified": True, "error": None}
    if deleted_call:
        return {"ok": False, "verified": False, "error": f"delete_unverified:{cid}"}
    error = "; ".join(method_errors) if method_errors else "no_supported_delete_method"
    return {"ok": False, "verified": False, "error": error}


def maybe_swap_current_art(art: Any, target_cid: str, fallback_ids: list[str]) -> tuple[bool, bool, Optional[str]]:
    current_id = extract_content_id(get_current_info(art))
    if current_id != target_cid:
        return True, False, None

    for fallback in fallback_ids:
        if not fallback or fallback == target_cid:
            continue
        try:
            art.select_image(fallback, show=True)
            for _ in range(3):
                time.sleep(0.6)
                after = extract_content_id(get_current_info(art))
                if after and after != target_cid:
                    return True, True, None
                art.select_image(fallback, show=True)
        except Exception as exc:
            continue
    return False, False, f"swap_failed_for_current:{target_cid}"


def collect_seed_fallback_ids(*catalog_paths: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in catalog_paths:
        catalog = load_frame_art_catalog(path)
        entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
        if not isinstance(entries, dict):
            continue
        for raw in entries.values():
            entry = seed_catalog_entry(raw)
            if entry.get("state") != "active":
                continue
            cid = str(entry.get("content_id", "")).strip()
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def cleanup_music_graph_for_deletion(catalog_key: str, content_id: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    changed = False
    key_name = Path(str(catalog_key or "")).name
    stem = music_catalog_stem(catalog_key)

    assoc = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    assoc_entries = assoc.get("entries") if isinstance(assoc.get("entries"), dict) else {}
    if isinstance(assoc_entries, dict):
        remove_keys: list[str] = []
        for alias, raw_record in assoc_entries.items():
            if not isinstance(raw_record, dict):
                continue
            rec_catalog_key = str(raw_record.get("catalog_key", "")).strip()
            rec_content_id = str(raw_record.get("content_id", "")).strip()
            if rec_catalog_key == catalog_key or (content_id and rec_content_id == content_id):
                remove_keys.append(alias)
        for alias in remove_keys:
            assoc_entries.pop(alias, None)
            changed = True
        if remove_keys:
            assoc["entries"] = assoc_entries
            persist_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH, assoc)

    for index_path in MUSIC_INDEX_PATHS:
        catalog, is_valid = load_frame_art_catalog_with_validity(index_path)
        if not is_valid:
            continue
        entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
        if not isinstance(entries, dict):
            continue
        remove_index_keys: list[str] = []
        for idx_key, raw_item in entries.items():
            if not isinstance(raw_item, dict):
                continue
            candidate_names = index_item_candidate_catalog_names(str(idx_key), raw_item)
            item_content_id = str(raw_item.get("content_id", "")).strip()
            if content_id and item_content_id == content_id:
                remove_index_keys.append(str(idx_key))
                continue
            if key_name and key_name in candidate_names:
                remove_index_keys.append(str(idx_key))
                continue
            if stem and str(idx_key).strip() == stem:
                remove_index_keys.append(str(idx_key))
        if remove_index_keys:
            for idx_key in remove_index_keys:
                entries.pop(idx_key, None)
            catalog["entries"] = entries
            persist_frame_art_catalog(index_path, catalog)
            changed = True

    delete_candidates: list[Path] = []
    if catalog_key:
        delete_candidates.extend(
            [
                COMPRESSED_DIR / catalog_key,
                COMPRESSED_DIR / key_name,
                WIDESCREEN_DIR / catalog_key,
                WIDESCREEN_DIR / key_name,
            ]
        )
    if stem:
        delete_candidates.extend(
            [
                SOURCE_DIR / f"{stem}.jpg",
                SOURCE_DIR / f"{stem}.png",
                BACKGROUND_DIR / f"{stem}__3840x2160__background.png",
                WIDESCREEN_DIR / f"{stem}__3840x2160.jpg",
                WIDESCREEN_DIR / f"{stem}__3840x2160.png",
                COMPRESSED_DIR / f"{stem}__3840x2160.jpg",
            ]
        )
    for path in delete_candidates:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                changed = True
        except Exception as exc:
            errors.append(f"file_delete:{path.name}:{repr(exc)}")

    return changed and not errors, errors


def process_seed_pending_deletions(
    tv_ip: str,
    art: Any,
    *,
    seed_kind: str,
    catalog_path: Path,
    catalog: dict[str, Any],
    delete_limit: int,
) -> tuple[Any, dict[str, Any]]:
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries

    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, raw_entry in entries.items():
        entry = seed_catalog_entry(raw_entry)
        entries[key] = entry
        if entry.get("state") == "pending_delete":
            candidates.append((str(key), entry))

    fallback_ids = collect_seed_fallback_ids(AMBIENT_CATALOG_PATH, HOLIDAY_CATALOG_PATH, MUSIC_CATALOG_PATH)
    fallback_ids.extend([cid for cid in available_content_ids(art) if cid not in fallback_ids])

    processed = 0
    failed = 0
    skipped_current = 0
    swap_used = 0
    errors: list[str] = []
    changed = False

    for key, _entry in candidates[: max(1, int(delete_limit))]:
        entry = seed_catalog_entry(entries.get(key))
        target_cid = str(entry.get("content_id", "")).strip()

        swap_ok, used_swap, swap_error = maybe_swap_current_art(art, target_cid, fallback_ids)
        if used_swap:
            swap_used += 1
        if not swap_ok:
            skipped_current += 1
            failed += 1
            if swap_error:
                errors.append(f"{key}:{swap_error}")
            continue

        delete_result = delete_art_content_id(art, target_cid)
        if target_cid and not bool(delete_result.get("ok")):
            failed += 1
            errors.append(f"{key}:{delete_result.get('error')}")
            continue

        if seed_kind == "music_seed":
            _music_cleanup_ok, music_cleanup_errors = cleanup_music_graph_for_deletion(key, target_cid)
            if music_cleanup_errors:
                failed += 1
                errors.extend([f"{key}:{msg}" for msg in music_cleanup_errors])
                continue

        mark_catalog_entry_deleted(entries, key)
        processed += 1
        changed = True

    if changed:
        persist_frame_art_catalog(catalog_path, catalog)

    return art, {
        "deletion_candidates": len(candidates),
        "deletion_processed": processed,
        "deletion_failed": failed,
        "deletion_skipped_currently_displayed": skipped_current,
        "deletion_swap_fallback_used": swap_used,
        "deletion_errors": errors[:20],
    }


def parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def music_association_record_rank(record: dict[str, Any]) -> tuple[int, int, float]:
    verified_rank = 1 if bool(record.get("verified", False)) else 0
    quality = str(record.get("source_quality", "")).strip().lower()
    if quality == "trusted_cache":
        quality_rank = 2
    elif quality == "generated":
        quality_rank = 1
    else:
        quality_rank = 0
    updated_dt = parse_iso_timestamp(record.get("updated_at"))
    updated_rank = updated_dt.timestamp() if updated_dt else 0.0
    return verified_rank, quality_rank, updated_rank


def choose_music_association_record(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate if music_association_record_rank(candidate) >= music_association_record_rank(existing) else existing


def compact_music_association_entries(
    entries: dict[str, Any],
    *,
    now: Optional[datetime] = None,
    session_ttl_days: int = MUSIC_ASSOCIATION_SESSION_TTL_DAYS,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    if not isinstance(entries, dict):
        return {}, {"input_entries": 0, "output_entries": 0, "trimmed_entries": 0}
    # Backward-compatible args retained; session/shazam aliases are now permanent.
    _ = now, session_ttl_days
    grouped: dict[tuple[str, str, str, str, str, Any], dict[str, Any]] = {}

    for key, raw_record in entries.items():
        if not isinstance(raw_record, dict):
            continue

        record = dict(raw_record)
        record.setdefault("cache_key", "")
        record.setdefault("catalog_key", "")
        record.setdefault("content_id", "")
        record.setdefault("key_source", "unknown")
        record.setdefault("artist", "")
        record.setdefault("album", "")
        record.setdefault("collection_id", None)
        record.setdefault("verified", False)
        record.setdefault("source_quality", "generated")
        record.setdefault("updated_at", "")

        signature = (
            str(record.get("cache_key", "")).strip(),
            str(record.get("catalog_key", "")).strip(),
            str(record.get("content_id", "")).strip(),
            str(record.get("artist", "")).strip(),
            str(record.get("album", "")).strip(),
            record.get("collection_id"),
        )
        group = grouped.setdefault(signature, {"record": record, "session_keys": [], "shazam_keys": []})
        group["record"] = choose_music_association_record(group["record"], record)

        raw_key = str(key or "")
        if raw_key.startswith("session::"):
            group["session_keys"].append(raw_key)
        elif raw_key.startswith("shazam::"):
            group["shazam_keys"].append(raw_key)

    compacted: dict[str, dict[str, Any]] = {}
    for group in grouped.values():
        record = group["record"]

        def set_key(target_key: str) -> None:
            if not target_key:
                return
            existing = compacted.get(target_key)
            if isinstance(existing, dict):
                compacted[target_key] = choose_music_association_record(existing, record)
            else:
                compacted[target_key] = record

        artist = str(record.get("artist", "")).strip()
        album = str(record.get("album", "")).strip()
        album_norm = normalized_album_association(artist, album)
        cache_key = str(record.get("cache_key", "")).strip()

        if album_norm:
            set_key(f"album_norm::{album_norm}")
        if cache_key:
            set_key(f"cache::{cache_key}")

        if group["session_keys"]:
            set_key(group["session_keys"][-1])
        if group["shazam_keys"]:
            set_key(group["shazam_keys"][-1])

    stats = {
        "input_entries": len(entries),
        "output_entries": len(compacted),
        "trimmed_entries": max(0, len(entries) - len(compacted)),
    }
    return compacted, stats


def compact_music_associations_file(session_ttl_days: int = MUSIC_ASSOCIATION_SESSION_TTL_DAYS) -> dict[str, int]:
    catalog = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    compacted, stats = compact_music_association_entries(entries, session_ttl_days=session_ttl_days)
    catalog["entries"] = compacted
    persist_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH, catalog)
    return stats


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
    set_catalog_entry_active(entries, catalog_key, content_id)
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


def load_music_triage() -> dict[str, Any]:
    if not MUSIC_TRIAGE_PATH.exists():
        return {"version": 1, "updated_at": "", "issues": []}
    try:
        data = json.loads(MUSIC_TRIAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_at": "", "issues": []}
    if not isinstance(data, dict):
        return {"version": 1, "updated_at": "", "issues": []}
    issues = data.get("issues")
    if not isinstance(issues, list):
        issues = []
    data["version"] = 1
    data["issues"] = issues
    data.setdefault("updated_at", "")
    return data


def persist_music_triage(catalog: dict[str, Any]) -> None:
    catalog["version"] = 1
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    catalog.setdefault("issues", [])
    atomic_write_json(MUSIC_TRIAGE_PATH, catalog)


def append_music_triage_issue(entry: dict[str, Any]) -> str:
    catalog = load_music_triage()
    issues = catalog.get("issues")
    if not isinstance(issues, list):
        issues = []
        catalog["issues"] = issues
    issue_id = uuid.uuid4().hex[:12]
    payload = dict(entry)
    payload["id"] = issue_id
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    issues.append(payload)
    persist_music_triage(catalog)
    return issue_id


def load_music_overrides() -> dict[str, Any]:
    return load_frame_art_catalog(MUSIC_OVERRIDES_PATH)


def persist_music_overrides(catalog: dict[str, Any]) -> None:
    persist_frame_art_catalog(MUSIC_OVERRIDES_PATH, catalog)


def set_music_override_for_album(*, artist: str, album: str, catalog_key: str, reason: str) -> bool:
    album_norm = normalized_album_association(artist, album)
    if not album_norm:
        return False
    key = f"album_norm::{album_norm}"
    catalog = load_music_overrides()
    entries = catalog.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries
    entries[key] = {
        "artist": artist,
        "album": album,
        "catalog_key": catalog_key,
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    persist_music_overrides(catalog)
    return True


def lookup_music_override(artist: str, album: str) -> Optional[dict[str, Any]]:
    album_norm = normalized_album_association(artist, album)
    if not album_norm:
        return None
    catalog = load_music_overrides()
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    key = f"album_norm::{album_norm}"
    override = entries.get(key) if isinstance(entries.get(key), dict) else None
    if not isinstance(override, dict):
        return None

    catalog_key = str(override.get("catalog_key", "")).strip()
    if not catalog_key:
        return None
    content_id = lookup_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key) or ""
    return {
        "cache_key": music_catalog_stem(catalog_key),
        "catalog_key": catalog_key,
        "content_id": content_id,
        "artist": artist,
        "album": album,
        "match_confidence": 1.0,
        "match_source": "manual_override",
    }


def normalized_album_association(artist: str, album: str) -> str:
    merged = f"{artist} {album}".strip().lower()
    merged = re.sub(r"[^a-z0-9]+", " ", merged)
    return re.sub(r"\s+", " ", merged).strip()


def normalize_music_text(value: str) -> str:
    cleaned = str(value or "").lower()
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_music_text_loose(value: str) -> str:
    # Canonicalize minor wording and format noise so near-identical album metadata
    # can still resolve to the same cached item.
    stopwords = {
        "a",
        "an",
        "and",
        "the",
        "of",
        "feat",
        "featuring",
        "ft",
        "with",
        "deluxe",
        "edition",
        "remaster",
        "remastered",
        "version",
        "anniversary",
        "album",
        "single",
        "ep",
    }
    return " ".join(token for token in normalize_music_text(value).split() if token not in stopwords)


def music_text_equivalent(query: str, candidate: str) -> bool:
    query_norm = normalize_music_text(query)
    candidate_norm = normalize_music_text(candidate)
    if query_norm and query_norm == candidate_norm:
        return True
    query_loose = normalize_music_text_loose(query)
    candidate_loose = normalize_music_text_loose(candidate)
    return bool(query_loose and query_loose == candidate_loose)


def music_token_set(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "the",
        "of",
        "feat",
        "featuring",
        "ft",
        "with",
        "deluxe",
        "edition",
        "remaster",
        "remastered",
        "version",
        "anniversary",
    }
    return {token for token in normalize_music_text(value).split() if len(token) > 1 and token not in stopwords}


def music_similarity(query: str, candidate: str) -> float:
    query_norm = normalize_music_text(query)
    candidate_norm = normalize_music_text(candidate)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        return 0.93

    q_tokens = music_token_set(query_norm)
    c_tokens = music_token_set(candidate_norm)
    if q_tokens and c_tokens:
        overlap = len(q_tokens & c_tokens) / float(len(q_tokens | c_tokens))
    else:
        overlap = 0.0

    if rapidfuzz_fuzz is not None:
        ratio = rapidfuzz_fuzz.ratio(query_norm, candidate_norm) / 100.0
        token_ratio = rapidfuzz_fuzz.token_set_ratio(query_norm, candidate_norm) / 100.0
        return (0.4 * ratio) + (0.35 * token_ratio) + (0.25 * overlap)

    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    return (0.65 * ratio) + (0.35 * overlap)


def parse_music_catalog_slug(catalog_key: str) -> str:
    key = str(catalog_key or "").strip()
    if not key:
        return ""
    filename = Path(key).name
    stem = filename.rsplit(".", 1)[0]
    stem = re.sub(r"__\d+x\d+(?:__background)?$", "", stem)
    if stem.isdigit():
        return ""
    m = re.match(r"^aa_(.+)_[0-9a-f]{8}$", stem)
    if m:
        return m.group(1).replace("-", " ")
    if stem.startswith("itc_"):
        return ""
    return stem.replace("-", " ")


def music_catalog_stem(catalog_key: str) -> str:
    filename = Path(str(catalog_key or "")).name
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"__\d+x\d+(?:__background)?$", "", stem)


def is_numeric_catalog_key(catalog_key: str) -> bool:
    return music_catalog_stem(catalog_key).isdigit()


def is_aa_catalog_key(catalog_key: str, cache_key: str = "") -> bool:
    stem = music_catalog_stem(catalog_key)
    if stem.startswith("aa_"):
        return True
    return str(cache_key or "").startswith("aa_")


def parse_collection_id_value(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def music_candidate_adjusted_score(
    score: float,
    *,
    catalog_key: str,
    cache_key: str = "",
    verified: bool = False,
) -> float:
    adjusted = score
    if is_aa_catalog_key(catalog_key, cache_key):
        adjusted -= 0.08
    if is_numeric_catalog_key(catalog_key):
        adjusted += 0.02
    if verified:
        adjusted += 0.03
    return max(0.0, min(1.0, adjusted))


def load_music_index_entries() -> dict[str, Any]:
    for path in MUSIC_INDEX_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        entries = payload.get("entries")
        if isinstance(entries, dict):
            return entries
    return {}


def load_music_manifest_entries() -> dict[str, Any]:
    for path in MUSIC_MANIFEST_PATHS:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                entries: dict[str, Any] = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("text_key") or item.get("collection_id") or item.get("collectionId") or "").strip()
                    if key:
                        entries[key] = item
                if entries:
                    return entries
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        manifest_entries = payload.get("entries")
        if isinstance(manifest_entries, dict):
            return manifest_entries
        return payload
    return {}


def resolve_music_source_art_path(raw_path: str) -> Optional[Path]:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists() and p.is_file():
        return p

    rel = Path(raw.lstrip("/"))
    base_roots = [
        Path("/media/frame_ai/music"),
        Path("/root/media/frame_ai/music"),
        Path("/media/frame_ai"),
        Path("/root/media/frame_ai"),
    ]
    candidates: list[Path] = []
    for root in base_roots:
        candidates.append(root / rel)
        candidates.append(root / rel.name)

    marker = "/out/source_art/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        for root in (Path("/media/frame_ai/music/out/source_art"), Path("/root/media/frame_ai/music/out/source_art")):
            candidates.append(root / suffix)
    marker2 = "/source_art/"
    if marker2 in raw:
        suffix2 = raw.split(marker2, 1)[1]
        for root in (Path("/media/frame_ai/music/source_art"), Path("/root/media/frame_ai/music/source_art")):
            candidates.append(root / suffix2)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def select_index_item_for_source(
    *,
    artist: str,
    album: str,
    collection_id: Optional[int],
) -> Optional[dict[str, Any]]:
    text_key_query = music_text_key(artist, album)
    for entries in (load_music_index_entries(), load_music_manifest_entries()):
        if not isinstance(entries, dict):
            continue
        if isinstance(collection_id, int):
            item = entries.get(str(collection_id))
            if isinstance(item, dict):
                return item
            for record in entries.values():
                if not isinstance(record, dict):
                    continue
                rec_cid = parse_collection_id_value(record.get("collection_id"))
                if rec_cid is None:
                    rec_cid = parse_collection_id_value(record.get("collectionId"))
                if rec_cid == collection_id:
                    return record
        if text_key_query:
            for record in entries.values():
                if not isinstance(record, dict):
                    continue
                text_key = str(record.get("text_key", "")).strip()
                if text_key and music_text_equivalent(text_key_query, text_key):
                    return record
    return None


def maybe_stage_source_art_from_index(
    *,
    src_path: Path,
    artist: str,
    album: str,
    collection_id: Optional[int],
) -> Optional[Path]:
    item = select_index_item_for_source(artist=artist, album=album, collection_id=collection_id)
    if not isinstance(item, dict):
        return None
    for key in ("source_art_path", "source_path", "source_album_path"):
        raw = str(item.get(key, "")).strip()
        candidate = resolve_music_source_art_path(raw)
        if candidate is None:
            continue
        src_path.parent.mkdir(parents=True, exist_ok=True)
        if candidate != src_path:
            shutil.copy2(candidate, src_path)
        log_event(
            "music_source_staged_from_index",
            source_key=key,
            source_path=str(candidate),
            staged_path=str(src_path),
            collection_id=collection_id,
            text_key=str(item.get("text_key", "")).strip(),
        )
        return candidate
    return None


def find_music_file_candidate_by_name(filename: str) -> Optional[Path]:
    name = str(filename or "").strip()
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (COMPRESSED_DIR, WIDESCREEN_DIR):
        path = root / name
        if path.exists():
            return path
    return None


def index_item_candidate_catalog_names(index_key: str, index_item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        n = str(name or "").strip()
        if not n or n in seen:
            return
        seen.add(n)
        names.append(n)

    compressed_path = str(index_item.get("compressed_output_path", "")).strip()
    output_path = str(index_item.get("output_path", "")).strip()
    if compressed_path:
        add(Path(compressed_path).name)
    if output_path:
        add(Path(output_path).name)

    key = str(index_key or "").strip()
    if key:
        stem = re.sub(r"__\d+x\d+(?:__background)?$", "", key)
        if stem:
            add(f"{stem}__3840x2160.jpg")
            add(f"{stem}__3840x2160.png")
    return names


def find_exact_music_index_match(artist: str, album: str) -> Optional[dict[str, Any]]:
    query_text_key = music_text_key(artist, album)
    if not normalize_music_text(query_text_key):
        return None

    index_entries = load_music_index_entries()
    if not isinstance(index_entries, dict):
        return None

    for index_key, index_item in index_entries.items():
        if not isinstance(index_item, dict):
            continue
        text_key = str(index_item.get("text_key", "")).strip()
        if not text_key:
            continue
        if not music_text_equivalent(query_text_key, text_key):
            continue

        catalog_key = ""
        for name in index_item_candidate_catalog_names(str(index_key), index_item):
            if find_music_file_candidate_by_name(name) is not None:
                catalog_key = name
                break
            if not catalog_key:
                catalog_key = name

        if not catalog_key:
            continue

        record: dict[str, Any] = {
            "cache_key": Path(catalog_key).name.rsplit(".", 1)[0],
            "catalog_key": catalog_key,
            "content_id": str(index_item.get("content_id", "")).strip(),
            "artist": artist,
            "album": album,
            "match_confidence": 1.0,
            "match_source": "index_text_key_exact",
        }
        stem = Path(catalog_key).name.rsplit(".", 1)[0]
        stem = re.sub(r"__\d+x\d+(?:__background)?$", "", stem)
        if stem.isdigit():
            try:
                record["collection_id"] = int(stem)
            except Exception:
                pass
        return record
    return None


def find_music_match_by_collection_id(collection_id: Optional[int]) -> Optional[dict[str, Any]]:
    if not isinstance(collection_id, int):
        return None

    music_catalog = load_frame_art_catalog(MUSIC_CATALOG_PATH)
    catalog_entries = music_catalog.get("entries") if isinstance(music_catalog.get("entries"), dict) else {}
    index_entries = load_music_index_entries()
    collection_key = str(collection_id)

    candidate_names = [f"{collection_key}__3840x2160.jpg", f"{collection_key}__3840x2160.png"]
    if isinstance(index_entries, dict):
        index_item = index_entries.get(collection_key)
        if isinstance(index_item, dict):
            for name in index_item_candidate_catalog_names(collection_key, index_item):
                if name not in candidate_names:
                    candidate_names.append(name)

    for name in candidate_names:
        entry = catalog_entries.get(name) if isinstance(catalog_entries, dict) and isinstance(catalog_entries.get(name), dict) else {}
        content_id = str(entry.get("content_id", "")).strip() if entry else ""
        if not content_id and isinstance(index_entries, dict):
            index_item = index_entries.get(collection_key)
            if isinstance(index_item, dict):
                content_id = str(index_item.get("content_id", "")).strip()
        if entry or find_music_file_candidate_by_name(name):
            return {
                "cache_key": music_catalog_stem(name),
                "catalog_key": name,
                "content_id": content_id,
                "collection_id": collection_id,
                "match_confidence": 1.0,
                "match_source": "collection_id_catalog_exact",
            }
    return None


def music_text_key(artist: str, album: str) -> str:
    artist_norm = normalize_music_text(artist)
    album_norm = normalize_music_text(album)
    if artist_norm and album_norm:
        return f"{artist_norm} — {album_norm}"
    return artist_norm or album_norm


def music_index_entry_key(*, collection_id: Optional[int], catalog_key: str, cache_key: str) -> str:
    if isinstance(collection_id, int):
        return str(collection_id)
    stem = Path(catalog_key).name.rsplit(".", 1)[0] if catalog_key else ""
    stem = re.sub(r"__\d+x\d+(?:__background)?$", "", stem)
    return stem or cache_key


def select_music_index_write_path() -> Path:
    for path in MUSIC_INDEX_PATHS:
        parent = path.parent
        if parent.exists() and parent.is_dir():
            return path
    return MUSIC_INDEX_PATHS[0]


def update_music_index_entry(
    *,
    artist: str,
    album: str,
    collection_id: Optional[int],
    catalog_key: str,
    cache_key: str,
    content_id: str,
    wide_path: Path,
    compressed_path: Path,
    request_id: Optional[str],
) -> None:
    index_path = select_music_index_write_path()
    catalog, is_valid = load_frame_art_catalog_with_validity(index_path)
    if not is_valid:
        log_event(
            "music_index_update_skipped_invalid_json",
            index_path=str(index_path),
        )
        return
    entries = catalog.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries

    entry_key = music_index_entry_key(collection_id=collection_id, catalog_key=catalog_key, cache_key=cache_key)
    if not entry_key:
        return

    existing = entries.get(entry_key) if isinstance(entries.get(entry_key), dict) else {}
    text_key = music_text_key(artist, album)
    payload = dict(existing)
    if text_key:
        payload["text_key"] = text_key
    payload["status"] = "ok"
    payload["content_id"] = str(content_id or "").strip()
    payload["prompt_variant"] = "reference_background_nomask"
    payload["output_path"] = str(wide_path)
    payload["compressed_output_path"] = str(compressed_path)
    if request_id:
        payload["request_id"] = request_id
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    entries[entry_key] = payload
    persist_frame_art_catalog(index_path, catalog)


def lookup_music_association_fuzzy(restore_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    artist = str(restore_payload.get("artist", "")).strip()
    album = str(restore_payload.get("album", "")).strip()
    query_album_artist = normalize_music_text(f"{artist} {album}".strip())
    if not (artist or album or query_album_artist):
        return None

    collection_id = parse_collection_id_value(restore_payload.get("collection_id"))
    collection_match = find_music_match_by_collection_id(collection_id)
    if isinstance(collection_match, dict):
        collection_match["artist"] = artist
        collection_match["album"] = album
        return collection_match

    exact_index_match = find_exact_music_index_match(artist, album)
    if isinstance(exact_index_match, dict):
        return exact_index_match

    candidates: list[dict[str, Any]] = []

    assoc_catalog = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    assoc_entries = assoc_catalog.get("entries") if isinstance(assoc_catalog.get("entries"), dict) else {}
    if isinstance(assoc_entries, dict):
        seen_keys: set[str] = set()
        for record in assoc_entries.values():
            if not isinstance(record, dict):
                continue
            dedupe_key = "|".join(
                [
                    str(record.get("cache_key", "")).strip(),
                    str(record.get("catalog_key", "")).strip(),
                    str(record.get("content_id", "")).strip(),
                ]
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            rec_artist = str(record.get("artist", "")).strip()
            rec_album = str(record.get("album", "")).strip()
            rec_combo = normalize_music_text(f"{rec_artist} {rec_album}".strip())

            artist_score = music_similarity(artist, rec_artist) if artist and rec_artist else 0.0
            album_score = music_similarity(album, rec_album) if album and rec_album else 0.0
            combo_score = music_similarity(query_album_artist, rec_combo) if query_album_artist and rec_combo else 0.0
            raw_score = max(combo_score, (0.5 * artist_score) + (0.5 * album_score))
            catalog_key = str(record.get("catalog_key", "")).strip()
            cache_key = str(record.get("cache_key", "")).strip()
            verified = bool(record.get("verified", False))
            score = music_candidate_adjusted_score(
                raw_score,
                catalog_key=catalog_key,
                cache_key=cache_key,
                verified=verified,
            )
            payload = dict(record)
            payload["match_source"] = "association_fuzzy"
            payload["match_confidence_raw"] = round(raw_score, 4)
            payload["match_confidence"] = round(score, 4)
            candidates.append(payload)

    music_catalog = load_frame_art_catalog(MUSIC_CATALOG_PATH)
    catalog_entries = music_catalog.get("entries") if isinstance(music_catalog.get("entries"), dict) else {}
    index_entries = load_music_index_entries()
    text_key_by_catalog_name: dict[str, str] = {}
    if isinstance(index_entries, dict):
        for collection_id, index_item in index_entries.items():
            if not isinstance(index_item, dict):
                continue
            text_key = str(index_item.get("text_key", "")).strip()
            if not text_key:
                continue
            if isinstance(collection_id, str) and collection_id.strip():
                cid = collection_id.strip()
                text_key_by_catalog_name[f"{cid}__3840x2160.jpg"] = text_key
                text_key_by_catalog_name[f"{cid}__3840x2160.png"] = text_key
            compressed_path = str(index_item.get("compressed_output_path", "")).strip()
            if compressed_path:
                text_key_by_catalog_name[Path(compressed_path).name] = text_key
            output_path = str(index_item.get("output_path", "")).strip()
            if output_path:
                text_key_by_catalog_name[Path(output_path).name] = text_key

    if isinstance(catalog_entries, dict):
        for catalog_key, entry in catalog_entries.items():
            if not isinstance(catalog_key, str):
                continue
            filename = Path(catalog_key).name
            stem = filename.rsplit(".", 1)[0]
            stem = re.sub(r"__\d+x\d+(?:__background)?$", "", stem)
            slug_text = parse_music_catalog_slug(catalog_key)
            candidate_text = slug_text
            source = "catalog_filename_fuzzy"
            if not candidate_text:
                index_text_key = text_key_by_catalog_name.get(filename, "")
                if index_text_key:
                    candidate_text = index_text_key
                    source = "catalog_index_text_key"
            if not candidate_text:
                continue
            raw_score = music_similarity(query_album_artist, candidate_text)
            score = music_candidate_adjusted_score(
                raw_score,
                catalog_key=catalog_key,
                cache_key=stem,
                verified=is_numeric_catalog_key(catalog_key),
            )
            content_id = ""
            if isinstance(entry, dict):
                content_id = str(entry.get("content_id", "")).strip()
            payload = {
                "cache_key": stem,
                "catalog_key": catalog_key,
                "content_id": content_id,
                "artist": artist,
                "album": album,
                "match_source": source,
                "match_confidence_raw": round(raw_score, 4),
                "match_confidence": round(score, 4),
            }
            if stem.isdigit():
                try:
                    payload["collection_id"] = int(stem)
                except Exception:
                    pass
            candidates.append(payload)

    if not candidates:
        return None

    candidates.sort(key=lambda c: float(c.get("match_confidence", 0.0)), reverse=True)
    best_record = dict(candidates[0])
    best_score = float(best_record.get("match_confidence", 0.0))
    second_best_score = float(candidates[1].get("match_confidence", 0.0)) if len(candidates) > 1 else 0.0

    top_candidates = []
    for item in candidates[:3]:
        top_candidates.append(
            {
                "source": item.get("match_source"),
                "catalog_key": item.get("catalog_key"),
                "score": item.get("match_confidence"),
                "score_raw": item.get("match_confidence_raw"),
            }
        )

    if best_score >= 0.78 and (best_score - second_best_score >= 0.05 or second_best_score == 0.0):
        best_record["match_candidates"] = top_candidates
        return best_record

    if best_score >= 0.74 and second_best_score >= 0.70 and (best_score - second_best_score) < 0.05:
        return {
            "generation_blocked": True,
            "match_source": "fuzzy_ambiguous_blocked",
            "match_confidence": round(best_score, 4),
            "second_match_confidence": round(second_best_score, 4),
            "match_candidates": top_candidates,
            "artist": artist,
            "album": album,
        }

    return None


def lookup_music_association(restore_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    collection_id = parse_collection_id_value(restore_payload.get("collection_id"))
    collection_match = find_music_match_by_collection_id(collection_id)
    if isinstance(collection_match, dict):
        collection_match["artist"] = str(restore_payload.get("artist", "")).strip()
        collection_match["album"] = str(restore_payload.get("album", "")).strip()
        return collection_match

    artist = str(restore_payload.get("artist", "")).strip()
    album = str(restore_payload.get("album", "")).strip()
    override_match = lookup_music_override(artist, album)
    if isinstance(override_match, dict):
        return override_match

    catalog = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    if not isinstance(entries, dict) or not entries:
        return lookup_music_association_fuzzy(restore_payload)

    music_session_key = str(restore_payload.get("music_session_key", "")).strip()
    shazam_key = str(restore_payload.get("shazam_key", "")).strip()
    album_key_raw = (artist + " — " + album).strip()
    album_key_norm = normalized_album_association(artist, album)

    candidate_keys: list[str] = []
    if music_session_key:
        candidate_keys.append(f"session::{music_session_key}")
    if shazam_key:
        candidate_keys.append(f"shazam::{shazam_key}")
    if album_key_raw:
        candidate_keys.append(f"album::{album_key_raw}")
    if album_key_norm:
        candidate_keys.append(f"album_norm::{album_key_norm}")
    album_loose = normalize_music_text_loose(f"{artist} {album}".strip())
    if album_loose:
        candidate_keys.append(f"album_loose::{album_loose}")

    for key in candidate_keys:
        record = entries.get(key)
        if isinstance(record, dict):
            return record

    if album_key_norm:
        for key, record in entries.items():
            if not (isinstance(key, str) and key.startswith("album::") and isinstance(record, dict)):
                continue
            raw_album_key = key.split("::", 1)[1]
            raw_norm = normalized_album_association(*raw_album_key.split(" — ", 1)) if " — " in raw_album_key else ""
            if raw_norm == album_key_norm:
                return record

    return lookup_music_association_fuzzy(restore_payload)


def update_music_association(
    restore_payload: dict[str, Any],
    *,
    cache_key: str,
    catalog_key: str,
    content_id: str,
    verified: Optional[bool] = None,
    source_quality: Optional[str] = None,
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
    collection_id = parse_collection_id_value(restore_payload.get("collection_id"))
    verified_flag = bool(is_numeric_catalog_key(catalog_key)) if verified is None else bool(verified)
    quality = source_quality or ("trusted_cache" if verified_flag else "generated")

    record = {
        "cache_key": cache_key,
        "catalog_key": catalog_key,
        "content_id": content_id,
        "key_source": key_source,
        "artist": artist,
        "album": album,
        "collection_id": collection_id if isinstance(collection_id, int) else None,
        "verified": verified_flag,
        "source_quality": quality,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if music_session_key:
        entries[f"session::{music_session_key}"] = record
    if shazam_key:
        entries[f"shazam::{shazam_key}"] = record
    album_norm = normalized_album_association(artist, album)
    if album_norm:
        entries[f"album_norm::{album_norm}"] = record
    if cache_key:
        entries[f"cache::{cache_key}"] = record

    compacted_entries, compact_stats = compact_music_association_entries(entries)
    catalog["entries"] = compacted_entries

    persist_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH, catalog)
    log_event("music_associations_compacted", **compact_stats)


def music_catalog_key_for_path(path: Path) -> str:
    try:
        return path.relative_to(COMPRESSED_DIR).as_posix()
    except Exception:
        return path.name


def handle_seed_restore(
    tv_ip: str,
    art: Any,
    restore_payload: dict,
    requested_at: str,
    *,
    seed_kind: str,
    seed_dir_key: str,
    default_seed_dir: Path,
    default_catalog_path: Path,
) -> dict[str, Any]:
    seed_dir_raw = str(restore_payload.get(seed_dir_key, "")).strip()
    catalog_path_raw = str(restore_payload.get("catalog_path", "")).strip()
    force_reupload = bool(restore_payload.get("force_reupload", False))
    apply_deletions = bool(restore_payload.get("apply_deletions", True))
    auto_queue_missing = bool(restore_payload.get("auto_queue_missing", True))
    try:
        delete_limit = max(1, int(restore_payload.get("delete_limit", DEFAULT_SEED_DELETE_LIMIT)))
    except Exception:
        delete_limit = DEFAULT_SEED_DELETE_LIMIT

    if not seed_dir_raw:
        seed_dir_raw = str(default_seed_dir)
    if not catalog_path_raw:
        catalog_path_raw = str(default_catalog_path)

    seed_dir = Path(seed_dir_raw)
    if not seed_dir.exists() or not seed_dir.is_dir():
        raise ValueError(f"{seed_kind} directory missing: {seed_dir}")

    files = list_seed_images(seed_dir)
    if not files:
        raise ValueError(f"{seed_kind} directory has no supported images: {seed_dir}")

    catalog_path = Path(catalog_path_raw)
    catalog = load_frame_art_catalog(catalog_path)
    entries = catalog.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        catalog["entries"] = entries

    existing_keys = {p.relative_to(seed_dir).as_posix() for p in files}
    existing_basenames = {p.name for p in files}
    auto_queued = 0
    if auto_queue_missing:
        for key, raw_entry in list(entries.items()):
            entry = seed_catalog_entry(raw_entry)
            entries[key] = entry
            cid = str(entry.get("content_id", "")).strip()
            is_legacy_basename_hit = "/" not in key and key in existing_basenames
            if entry.get("state") == "active" and cid and key not in existing_keys and not is_legacy_basename_hit:
                if mark_catalog_entry_pending_delete(entries, key, reason="missing_file_drift"):
                    auto_queued += 1
        if auto_queued:
            persist_frame_art_catalog(catalog_path, catalog)

    deletion_status = {
        "deletion_candidates": 0,
        "deletion_processed": 0,
        "deletion_failed": 0,
        "deletion_skipped_currently_displayed": 0,
        "deletion_swap_fallback_used": 0,
        "deletion_errors": [],
    }
    if apply_deletions:
        art, deletion_status = process_seed_pending_deletions(
            tv_ip,
            art,
            seed_kind=seed_kind,
            catalog_path=catalog_path,
            catalog=catalog,
            delete_limit=delete_limit,
        )

    uploaded_count = 0
    skipped_count = 0
    failed_count = 0
    failures: list[str] = []
    last_cid: Optional[str] = None

    for image_path in files:
        key = image_path.relative_to(seed_dir).as_posix()
        entry = seed_catalog_entry(entries.get(key))
        if entry.get("content_id", "") == "" and "/" in key:
            legacy_key = image_path.name
            legacy_entry = seed_catalog_entry(entries.get(legacy_key))
            if str(legacy_entry.get("content_id", "")).strip():
                entry = legacy_entry
        entries[key] = entry
        if entry.get("state") != "active":
            continue
        cached_id = str(entry.get("content_id", "")).strip()

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
            set_catalog_entry_active(entries, key, content_id)
            persist_frame_art_catalog(catalog_path, catalog)
        except Exception as exc:
            failed_count += 1
            failures.append(f"{key}: {repr(exc)}")

    ok = failed_count == 0 and deletion_status.get("deletion_failed", 0) == 0 and (
        uploaded_count > 0 or skipped_count > 0 or deletion_status.get("deletion_processed", 0) > 0
    )
    return {
        "ok": ok,
        "mode": "restore",
        "kind": seed_kind,
        "requested_at": requested_at,
        seed_dir_key: str(seed_dir),
        "catalog_path": str(catalog_path),
        "force_reupload": force_reupload,
        "apply_deletions": apply_deletions,
        "auto_queue_missing": auto_queue_missing,
        "delete_limit": delete_limit,
        "auto_queued_missing_count": auto_queued,
        "total_files": len(files),
        "uploaded_count": uploaded_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "selected_content_id": last_cid,
        "error": None if ok else "; ".join(failures) if failures else f"{seed_kind} completed with no successful uploads",
        **deletion_status,
    }


def handle_ambient_seed_restore(tv_ip: str, art: Any, restore_payload: dict, requested_at: str) -> dict[str, Any]:
    return handle_seed_restore(
        tv_ip,
        art,
        restore_payload,
        requested_at,
        seed_kind="ambient_seed",
        seed_dir_key="ambient_dir",
        default_seed_dir=Path("/media/frame_ai/ambient"),
        default_catalog_path=AMBIENT_CATALOG_PATH,
    )


def handle_holiday_seed_restore(tv_ip: str, art: Any, restore_payload: dict, requested_at: str) -> dict[str, Any]:
    return handle_seed_restore(
        tv_ip,
        art,
        restore_payload,
        requested_at,
        seed_kind="holiday_seed",
        seed_dir_key="holiday_dir",
        default_seed_dir=Path("/media/frame_ai/holidays"),
        default_catalog_path=HOLIDAY_CATALOG_PATH,
    )


def handle_music_seed_restore(tv_ip: str, art: Any, restore_payload: dict, requested_at: str) -> dict[str, Any]:
    return handle_seed_restore(
        tv_ip,
        art,
        restore_payload,
        requested_at,
        seed_kind="music_seed",
        seed_dir_key="music_dir",
        default_seed_dir=COMPRESSED_DIR,
        default_catalog_path=MUSIC_CATALOG_PATH,
    )


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


def enqueue_restore_payload(payload: dict[str, Any]) -> Path:
    RESTORE_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    unique = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json"
    queued = RESTORE_QUEUE_DIR / unique
    atomic_write_json(queued, payload)
    return queued


def enqueue_music_feedback_item(payload: dict[str, Any]) -> Path:
    MUSIC_FEEDBACK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    unique = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json"
    queued = MUSIC_FEEDBACK_QUEUE_DIR / unique
    atomic_write_json(queued, payload)
    return queued


def resolve_music_catalog_path(catalog_key: str) -> Optional[Path]:
    key = Path(str(catalog_key or "").strip()).name
    if not key:
        return None
    for root in (COMPRESSED_DIR, WIDESCREEN_DIR):
        candidate = root / key
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def build_music_request_from_feedback(payload: dict[str, Any], *, show: bool, force_regen: bool = False) -> dict[str, Any]:
    return {
        "kind": "cover_art_reference_background",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "show": show,
        "music_session_key": str(payload.get("music_session_key", "")).strip(),
        "listening_mode": str(payload.get("listening_mode", "")).strip(),
        "artist": str(payload.get("artist", "")).strip(),
        "album": str(payload.get("album", "")).strip(),
        "track": str(payload.get("track", "")).strip(),
        "key_source": str(payload.get("key_source", "")).strip().lower(),
        "shazam_key": str(payload.get("shazam_key", "")).strip(),
        "collection_id": parse_collection_id_value(payload.get("collection_id")),
        "source_preference": "itunes",
        "force_regen": force_regen,
    }


def dequeue_next_restore_work_item() -> Optional[Path]:
    queued = list_queued_requests()
    if not queued:
        return None
    return queued[0]


def is_music_restore_kind(kind: str) -> bool:
    return str(kind or "").strip().lower() in MUSIC_RESTORE_KINDS


def is_superseded_music_request(work_item: Path, current_kind: str) -> bool:
    if not is_music_restore_kind(current_kind):
        return False

    queued = list_queued_requests()
    if not queued:
        return False

    current_index = -1
    for idx, queued_item in enumerate(queued):
        if queued_item.name == work_item.name:
            current_index = idx
            break
    if current_index < 0:
        return False

    for queued_item in queued[current_index + 1 :]:
        next_payload, next_requested_show, next_parse_error = load_restore_work_item(queued_item)
        if next_parse_error or not isinstance(next_payload, dict):
            continue
        next_kind = str(next_payload.get("kind", "")).strip().lower()
        if is_music_restore_kind(next_kind) and next_requested_show is not False:
            return True

    return False


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


def is_art_socket_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (BrokenPipeError, TimeoutError)):
        return True
    message = repr(exc).lower()
    return (
        "broken pipe" in message
        or "timed out" in message
        or "timeout" in message
        or "websocket time out" in message
        or "connection timed out" in message
    )


def resolve_runtime_int_option(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        raw = int(RUNTIME_OPTIONS.get(name, default))
    except Exception:
        raw = default
    return max(min_value, min(max_value, raw))


def retry_backoff_seconds(attempt_index: int) -> float:
    base = resolve_runtime_int_option("art_retry_backoff_s", 2, min_value=1, max_value=10)
    return float(min(base * (attempt_index + 1), 20))


def select_and_verify_with_reconnect(
    tv_ip: str,
    art: Any,
    target_cid: str,
    attempts: int = 3,
    sleep_s: float = 1.0,
) -> tuple[Any, bool, Optional[Exception]]:
    select_error: Optional[Exception] = None
    current_art = art
    socket_attempts = resolve_runtime_int_option("art_socket_retries", 5, min_value=1, max_value=10)

    for socket_attempt in range(socket_attempts):
        try:
            current_art.select_image(target_cid, show=True)
            for _ in range(max(1, attempts)):
                time.sleep(sleep_s)
                current_info = get_current_info(current_art)
                current_id = extract_content_id(current_info)
                if current_id == target_cid:
                    return current_art, True, None
                current_art.select_image(target_cid, show=True)
            return current_art, False, None
        except Exception as exc:
            select_error = exc
            if socket_attempt < socket_attempts - 1 and is_art_socket_retryable_error(exc):
                log_event(
                    "select_retry",
                    reason=repr(exc),
                    selected_content_id=target_cid,
                    action="reconnect_art_socket",
                    attempt=socket_attempt + 1,
                    max_attempts=socket_attempts,
                )
                time.sleep(retry_backoff_seconds(socket_attempt))
                try:
                    retry_tv = SamsungTVWS(tv_ip)
                    retry_art = retry_tv.art()
                    if not retry_art.supported():
                        select_error = RuntimeError("Art mode not supported / unreachable")
                        break
                    current_art = retry_art
                    continue
                except Exception as reconnect_exc:
                    select_error = reconnect_exc
                    log_event(
                        "select_retry_reconnect_failed",
                        reason=repr(reconnect_exc),
                        selected_content_id=target_cid,
                        attempt=socket_attempt + 1,
                        max_attempts=socket_attempts,
                    )
                    if is_art_socket_retryable_error(reconnect_exc):
                        continue
                    break
            break

    return current_art, False, select_error


def invalidate_music_cached_content_id(catalog_key: str, content_id: str) -> None:
    key_name = Path(str(catalog_key or "")).name
    target_cid = str(content_id or "").strip()
    if not key_name and not target_cid:
        return

    catalog = load_frame_art_catalog(MUSIC_CATALOG_PATH)
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), dict) else {}
    if not isinstance(entries, dict):
        entries = {}
    if key_name:
        entry = seed_catalog_entry(entries.get(key_name))
        if not target_cid or str(entry.get("content_id", "")).strip() == target_cid:
            entry["content_id"] = ""
            entries[key_name] = entry
            catalog["entries"] = entries
            persist_frame_art_catalog(MUSIC_CATALOG_PATH, catalog)

    assoc = load_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH)
    assoc_entries = assoc.get("entries") if isinstance(assoc.get("entries"), dict) else {}
    assoc_changed = False
    if isinstance(assoc_entries, dict):
        for alias, raw_record in assoc_entries.items():
            if not isinstance(raw_record, dict):
                continue
            rec_catalog_key = str(raw_record.get("catalog_key", "")).strip()
            rec_content_id = str(raw_record.get("content_id", "")).strip()
            if (key_name and rec_catalog_key == key_name) or (target_cid and rec_content_id == target_cid):
                raw_record["content_id"] = ""
                raw_record["verified"] = False
                raw_record["updated_at"] = datetime.now(timezone.utc).isoformat()
                assoc_entries[alias] = raw_record
                assoc_changed = True
    if assoc_changed:
        assoc["entries"] = assoc_entries
        persist_frame_art_catalog(MUSIC_ASSOCIATIONS_PATH, assoc)


def upload_local_file_with_reconnect(tv_ip: str, art: Any, file_path: Path) -> tuple[Any, Optional[str]]:
    current_art = art
    upload_attempts = resolve_runtime_int_option("art_socket_retries", 5, min_value=1, max_value=10)
    last_exc: Optional[Exception] = None
    for attempt in range(upload_attempts):
        try:
            return current_art, upload_local_file(current_art, file_path)
        except Exception as exc:
            if not is_art_socket_retryable_error(exc):
                raise
            last_exc = exc
            if attempt >= upload_attempts - 1:
                break
            log_event(
                "upload_retry",
                reason=repr(exc),
                file=str(file_path),
                action="reconnect_art_socket",
                attempt=attempt + 1,
                max_attempts=upload_attempts,
            )
            time.sleep(retry_backoff_seconds(attempt))
            try:
                retry_tv = SamsungTVWS(tv_ip)
                retry_art = retry_tv.art()
                if not retry_art.supported():
                    last_exc = RuntimeError("Art mode not supported / unreachable")
                    break
                current_art = retry_art
            except Exception as reconnect_exc:
                last_exc = reconnect_exc
                log_event(
                    "upload_retry_reconnect_failed",
                    reason=repr(reconnect_exc),
                    file=str(file_path),
                    attempt=attempt + 1,
                    max_attempts=upload_attempts,
                )
                if not is_art_socket_retryable_error(reconnect_exc):
                    raise
                continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("upload_local_file_with_reconnect failed without an explicit exception")


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
    openai_model = str(opts.get("openai_model", "gpt-image-1.5")).strip() or "gpt-image-1.5"
    openai_timeout_s = resolve_runtime_int_option("openai_timeout_s", 120, min_value=30, max_value=600)
    _ = resolve_runtime_int_option("art_socket_retries", 5, min_value=1, max_value=10)
    _ = resolve_runtime_int_option("art_retry_backoff_s", 2, min_value=1, max_value=10)

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
                artist = ""
                album = ""
                track = ""
                match_candidates_for_status: list[dict[str, Any]] = []
                match_source_for_status: Optional[str] = None
                match_confidence_for_status: Optional[float] = None
                second_match_confidence_for_status: Optional[float] = None

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
                elif kind in SEED_KINDS:
                    if kind == "ambient_seed":
                        seed_status = handle_ambient_seed_restore(tv_ip, art, restore_payload, requested_at)
                    elif kind == "holiday_seed":
                        seed_status = handle_holiday_seed_restore(tv_ip, art, restore_payload, requested_at)
                    else:
                        seed_status = handle_music_seed_restore(tv_ip, art, restore_payload, requested_at)
                    write_status({**seed_status, "tv_ip": tv_ip, "addon_version": ADDON_VERSION})
                    continue
                elif kind == "music_feedback":
                    artist = str(restore_payload.get("artist", "")).strip()
                    album = str(restore_payload.get("album", "")).strip()
                    track = str(restore_payload.get("track", "")).strip()
                    action = str(restore_payload.get("action", "queue_only")).strip().lower() or "queue_only"
                    issue_type = str(restore_payload.get("issue_type", "")).strip().lower()
                    notes = str(restore_payload.get("notes", "")).strip()
                    requested_at = str(restore_payload.get("requested_at", "")).strip()
                    candidate_catalog_key = str(restore_payload.get("candidate_catalog_key", "")).strip()
                    current_content_id = str(restore_payload.get("current_content_id", "")).strip()
                    cache_key = str(restore_payload.get("cache_key", "")).strip()
                    issue_id = append_music_triage_issue(
                        {
                            "status": "open" if action == "queue_only" else "queued",
                            "action": action,
                            "issue_type": issue_type,
                            "artist": artist,
                            "album": album,
                            "track": track,
                            "music_session_key": str(restore_payload.get("music_session_key", "")).strip(),
                            "collection_id": parse_collection_id_value(restore_payload.get("collection_id")),
                            "current_content_id": current_content_id,
                            "cache_key": cache_key,
                            "candidate_catalog_key": candidate_catalog_key,
                            "notes": notes,
                        }
                    )
                    enqueue_music_feedback_item(
                        {
                            "issue_id": issue_id,
                            "requested_at": requested_at,
                            "action": action,
                            "issue_type": issue_type,
                            "artist": artist,
                            "album": album,
                            "track": track,
                            "cache_key": cache_key,
                            "candidate_catalog_key": candidate_catalog_key,
                            "current_content_id": current_content_id,
                            "notes": notes,
                        }
                    )

                    followup_kind = "none"
                    if action == "regen_now":
                        association_record = lookup_music_association(restore_payload)
                        if isinstance(association_record, dict):
                            stale_catalog_key = str(association_record.get("catalog_key", "")).strip()
                            stale_content_id = str(association_record.get("content_id", "")).strip()
                            if stale_catalog_key and stale_content_id:
                                invalidate_music_cached_content_id(stale_catalog_key, stale_content_id)
                        followup_payload = build_music_request_from_feedback(restore_payload, show=show_flag, force_regen=True)
                        enqueue_restore_payload(followup_payload)
                        followup_kind = "cover_art_reference_background"
                    elif action == "use_candidate_now":
                        if not candidate_catalog_key:
                            raise ValueError("music_feedback use_candidate_now requires candidate_catalog_key")
                        candidate_path = resolve_music_catalog_path(candidate_catalog_key)
                        if candidate_path is None:
                            raise ValueError(f"music_feedback candidate not found: {candidate_catalog_key}")
                        set_music_override_for_album(
                            artist=artist,
                            album=album,
                            catalog_key=Path(candidate_catalog_key).name,
                            reason=f"issue_type:{issue_type or 'manual'}",
                        )
                        followup_payload = build_music_request_from_feedback(restore_payload, show=show_flag, force_regen=False)
                        enqueue_restore_payload(followup_payload)
                        followup_kind = "cover_art_reference_background"

                    write_status(
                        {
                            "ok": True,
                            "mode": "restore",
                            "tv_ip": tv_ip,
                            "kind": kind,
                            "requested_at": requested_at,
                            "issue_id": issue_id,
                            "issue_type": issue_type,
                            "action": action,
                            "followup_kind": followup_kind,
                            "artist": artist,
                            "album": album,
                            "track": track,
                            "cache_key": cache_key,
                            "current_content_id": current_content_id,
                            "candidate_catalog_key": candidate_catalog_key,
                            "notes": notes,
                            "addon_version": ADDON_VERSION,
                        }
                    )
                    continue
                elif kind in {"cover_art_reference_background", "cover_art_outpaint"}:
                    ensure_dirs()
                    source_url = str(restore_payload.get("artwork_url", "")).strip()
                    artist = str(restore_payload.get("artist", "")).strip()
                    album = str(restore_payload.get("album", "")).strip()
                    force_regen = bool(restore_payload.get("force_regen", False))

                    def _maybe_float(value: Any) -> Optional[float]:
                        if value in (None, ""):
                            return None
                        try:
                            return float(value)
                        except Exception:
                            return None

                    association_record = lookup_music_association(restore_payload)
                    if isinstance(association_record, dict):
                        match_source_for_status = str(association_record.get("match_source", "exact"))
                        match_confidence_for_status = _maybe_float(association_record.get("match_confidence"))
                        second_match_confidence_for_status = _maybe_float(association_record.get("second_match_confidence"))
                        log_event(
                            "music_association_hit",
                            match_source=str(association_record.get("match_source", "exact")),
                            match_confidence=association_record.get("match_confidence"),
                            catalog_key=association_record.get("catalog_key"),
                            cache_key=association_record.get("cache_key"),
                            content_id=association_record.get("content_id"),
                        )
                        match_candidates = association_record.get("match_candidates")
                        if isinstance(match_candidates, list) and match_candidates:
                            match_candidates_for_status = [
                                c for c in match_candidates[:5] if isinstance(c, dict)
                            ]
                            log_event(
                                "music_match_candidates",
                                candidates=match_candidates,
                                blocked=bool(association_record.get("generation_blocked", False)),
                            )
                    collection_id_raw = restore_payload.get("collection_id")
                    collection_id: Optional[int] = None
                    if collection_id_raw not in (None, ""):
                        try:
                            collection_id = int(collection_id_raw)
                        except Exception:
                            collection_id = None
                    if collection_id is None and isinstance(association_record, dict):
                        assoc_collection_id = association_record.get("collection_id")
                        if isinstance(assoc_collection_id, int):
                            collection_id = assoc_collection_id

                    cache_key = normalize_key(collection_id, artist, album)
                    stem_key = str(collection_id) if isinstance(collection_id, int) else cache_key
                    association_cache_key = (
                        str(association_record.get("cache_key", "")).strip()
                        if isinstance(association_record, dict)
                        else ""
                    )
                    association_catalog_key = (
                        str(association_record.get("catalog_key", "")).strip()
                        if isinstance(association_record, dict)
                        else ""
                    )
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
                        if association_catalog_key:
                            association_compressed = COMPRESSED_DIR / association_catalog_key
                            association_wide = WIDESCREEN_DIR / association_catalog_key
                            for candidate in (association_compressed, association_wide):
                                if candidate.exists():
                                    wide_path = candidate
                                    break
                        association_legacy_png = (
                            WIDESCREEN_DIR / f"{association_cache_key}.png" if association_cache_key else None
                        )
                        association_legacy_jpg = (
                            WIDESCREEN_DIR / f"{association_cache_key}.jpg" if association_cache_key else None
                        )
                        for candidate in (wide_png_path, wide_jpg_path, legacy_png_path, legacy_jpg_path):
                            if candidate.exists():
                                wide_path = candidate
                                break
                        if not wide_path.exists():
                            for candidate in (association_legacy_png, association_legacy_jpg):
                                if candidate is not None and candidate.exists():
                                    wide_path = candidate
                                    break

                    resolved_folder = str(WIDESCREEN_DIR)
                    selected_name = wide_path.name
                    file_count = 1
                    chosen_index = 0
                    music_retry_upload_path: Optional[Path] = wide_path if wide_path.exists() else None
                    music_retry_catalog_key: Optional[str] = None

                    if force_regen:
                        selected_name = compressed_jpg_path.name
                        music_retry_upload_path = None
                        wide_path = Path("__force_regen__")

                    if wide_path.exists():
                        catalog_key = music_catalog_key_for_path(wide_path)
                        music_retry_catalog_key = catalog_key
                        cached_content_id = lookup_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key)
                        association_content_id = (
                            str(association_record.get("content_id", "")).strip()
                            if isinstance(association_record, dict)
                            else ""
                        )
                        if cached_content_id:
                            target_cid = cached_content_id
                            encoded_type = guess_file_type(wide_path)
                            encoded_bytes = wide_path.stat().st_size
                            update_music_association(
                                restore_payload,
                                cache_key=cache_key,
                                catalog_key=catalog_key,
                                content_id=target_cid,
                                verified=True,
                                source_quality="trusted_cache",
                            )
                            update_music_index_entry(
                                artist=artist,
                                album=album,
                                collection_id=collection_id,
                                catalog_key=catalog_key,
                                cache_key=cache_key,
                                content_id=target_cid,
                                wide_path=wide_path,
                                compressed_path=wide_path if wide_path.suffix.lower() in {".jpg", ".jpeg"} else compressed_jpg_path,
                                request_id=None,
                            )
                            pick_source = "music_cache_catalog_hit"
                        elif association_content_id:
                            target_cid = association_content_id
                            encoded_type = guess_file_type(wide_path)
                            encoded_bytes = wide_path.stat().st_size
                            update_catalog_content_id(MUSIC_CATALOG_PATH, catalog_key, target_cid)
                            update_music_association(
                                restore_payload,
                                cache_key=cache_key,
                                catalog_key=catalog_key,
                                content_id=target_cid,
                                verified=True,
                                source_quality="trusted_cache",
                            )
                            update_music_index_entry(
                                artist=artist,
                                album=album,
                                collection_id=collection_id,
                                catalog_key=catalog_key,
                                cache_key=cache_key,
                                content_id=target_cid,
                                wide_path=wide_path,
                                compressed_path=wide_path if wide_path.suffix.lower() in {".jpg", ".jpeg"} else compressed_jpg_path,
                                request_id=None,
                            )
                            pick_source = "music_cache_association_content_id"
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
                                verified=True,
                                source_quality="trusted_cache",
                            )
                            update_music_index_entry(
                                artist=artist,
                                album=album,
                                collection_id=collection_id,
                                catalog_key=catalog_key,
                                cache_key=cache_key,
                                content_id=target_cid,
                                wide_path=wide_path,
                                compressed_path=wide_path if wide_path.suffix.lower() in {".jpg", ".jpeg"} else compressed_jpg_path,
                                request_id=None,
                            )
                            pick_source = "music_cache_uploaded"
                    else:
                        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
                        cover_error: Optional[Exception] = None
                        fallback_path: Optional[Path] = None
                        try:
                            if isinstance(association_record, dict) and bool(association_record.get("generation_blocked", False)):
                                log_event(
                                    "music_generation_blocked",
                                    reason="ambiguous_match",
                                    match_source=association_record.get("match_source"),
                                    match_confidence=association_record.get("match_confidence"),
                                    second_match_confidence=association_record.get("second_match_confidence"),
                                    candidates=association_record.get("match_candidates"),
                                )
                                raise ValueError("Ambiguous music match; generation blocked to prevent duplicate cache entries")

                            if not src_path.exists():
                                maybe_stage_source_art_from_index(
                                    src_path=src_path,
                                    artist=artist,
                                    album=album,
                                    collection_id=collection_id,
                                )

                            if not source_url and not src_path.exists():
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
                                if not source_url and artist and album:
                                    album_candidates: list[str] = []

                                    def add_album_candidate(value: str) -> None:
                                        v = str(value or "").strip()
                                        if v and v not in album_candidates:
                                            album_candidates.append(v)

                                    add_album_candidate(album)
                                    add_album_candidate(re.sub(r"\s*\([^)]*\)\s*", " ", album))
                                    add_album_candidate(re.sub(r"\s*\[[^\]]*\]\s*", " ", album))
                                    if ":" in album:
                                        add_album_candidate(album.split(":", 1)[0])
                                        add_album_candidate(album.split(":", 1)[1])
                                    add_album_candidate(re.sub(r"\s*[-–]\s*expanded.*$", "", album, flags=re.IGNORECASE))

                                    for album_candidate in album_candidates:
                                        album_info = itunes_search(artist, album_candidate, timeout_s=10)
                                        source_url = resolve_artwork_url(album_info)
                                        if source_url:
                                            break
                                if not source_url and not (artist and album) and collection_id is None:
                                    raise ValueError("unsupported metadata; provide artwork_url, collection_id, or artist+album")

                            if not source_url and not src_path.exists():
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
                                    timeout_s=openai_timeout_s,
                                    album_shadow=True,
                                )
                                generation_mode = "openai_reference"
                            except Exception as gen_exc:
                                final_png, background_png = generate_local_fallback_frame_from_album(
                                    source_album_path=src_path,
                                    album_shadow=True,
                                )
                                request_id = None
                                model_used = "local-fallback"
                                gen_msg = repr(gen_exc).lower()
                                if "moderation_blocked" in gen_msg or "safety system" in gen_msg:
                                    generation_mode = "local_fallback_blocked"
                                else:
                                    generation_mode = "local_fallback_openai_error"
                                log_event(
                                    "music_openai_fallback",
                                    cache_key=cache_key,
                                    error=repr(gen_exc),
                                    mode=generation_mode,
                                )

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
                                verified=False,
                                source_quality="generated",
                            )
                            update_music_index_entry(
                                artist=artist,
                                album=album,
                                collection_id=collection_id,
                                catalog_key=catalog_key,
                                cache_key=cache_key,
                                content_id=target_cid,
                                wide_path=wide_path,
                                compressed_path=compressed_jpg_path if compressed_jpg_path.exists() else wide_path,
                                request_id=request_id,
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
                                    "track": str(restore_payload.get("track", "")).strip(),
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

                if show_flag and is_superseded_music_request(work_item, kind):
                    show_flag = False
                    log_event(
                        "restore_show_suppressed",
                        reason="superseded_by_newer_music_request",
                        kind=kind,
                        requested_at=requested_at,
                    )

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
                        art, verified, select_attempt_error = select_and_verify_with_reconnect(
                            tv_ip,
                            art,
                            target_cid,
                            attempts=3,
                            sleep_s=1.0,
                        )

                        retry_upload_file: Optional[Path] = pick_local_retry_file
                        retry_catalog_path: Optional[Path] = pick_local_retry_catalog_path
                        retry_catalog_key: Optional[str] = pick_local_retry_catalog_key
                        retry_event = "catalog_pick_fallback_upload"
                        if (
                            not verified
                            and retry_upload_file is None
                            and kind in {"cover_art_reference_background", "cover_art_outpaint"}
                            and music_retry_upload_path is not None
                        ):
                            retry_upload_file = music_retry_upload_path
                            retry_catalog_path = MUSIC_CATALOG_PATH
                            retry_catalog_key = music_retry_catalog_key or music_catalog_key_for_path(music_retry_upload_path)
                            retry_event = "music_catalog_fallback_upload"
                            log_event(
                                "music_catalog_invalidate_stale_content_id",
                                catalog_key=retry_catalog_key,
                                stale_content_id=target_cid,
                            )
                            invalidate_music_cached_content_id(retry_catalog_key or "", target_cid or "")

                        if not verified and retry_upload_file is not None:
                            log_event(
                                retry_event,
                                file=str(retry_upload_file),
                                selected_content_id=target_cid,
                                reason=repr(select_attempt_error) if select_attempt_error else "select_unverified",
                            )
                            art, replacement_cid = upload_local_file_with_reconnect(tv_ip, art, retry_upload_file)
                            if not replacement_cid:
                                raise ValueError(
                                    f"Upload fallback completed but content_id was not discovered (kind={kind}, value={request_value!r}, resolved_folder={resolved_folder})"
                                )
                            if kind in {"cover_art_reference_background", "cover_art_outpaint"}:
                                append_uploaded_id(state, "cover_uploaded_ids", replacement_cid)
                            else:
                                append_local_uploaded_id(state, replacement_cid)
                            target_cid = replacement_cid
                            if retry_catalog_path and retry_catalog_key:
                                update_catalog_content_id(
                                    retry_catalog_path,
                                    retry_catalog_key,
                                    replacement_cid,
                                )
                            if kind in {"cover_art_reference_background", "cover_art_outpaint"}:
                                update_music_association(
                                    restore_payload,
                                    cache_key=cache_key,
                                    catalog_key=retry_catalog_key or "",
                                    content_id=replacement_cid,
                                    verified=True,
                                    source_quality="trusted_cache",
                                )
                                update_music_index_entry(
                                    artist=artist,
                                    album=album,
                                    collection_id=collection_id,
                                    catalog_key=retry_catalog_key or "",
                                    cache_key=cache_key,
                                    content_id=replacement_cid,
                                    wide_path=retry_upload_file,
                                    compressed_path=retry_upload_file if retry_upload_file.suffix.lower() in {".jpg", ".jpeg"} else compressed_jpg_path,
                                    request_id=locals().get("request_id"),
                                )
                                pick_source = "music_cache_refreshed_upload"
                            art, verified, retry_select_error = select_and_verify_with_reconnect(
                                tv_ip,
                                art,
                                target_cid,
                                attempts=3,
                                sleep_s=1.0,
                            )
                            if retry_select_error is not None and not verified:
                                select_attempt_error = retry_select_error
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
                        "effective_show": show_flag,
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
                        "match_source": match_source_for_status,
                        "match_confidence": match_confidence_for_status,
                        "second_match_confidence": second_match_confidence_for_status,
                        "match_candidates": match_candidates_for_status,
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
                        "match_source": locals().get("match_source_for_status"),
                        "match_confidence": locals().get("match_confidence_for_status"),
                        "second_match_confidence": locals().get("second_match_confidence_for_status"),
                        "match_candidates": locals().get("match_candidates_for_status", []),
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
