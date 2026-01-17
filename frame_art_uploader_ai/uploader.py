import json
import re
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from samsungtvws import SamsungTVWS

OPTIONS_PATH = "/data/options.json"
STATUS_PATH = "/share/frame_art_uploader_last.json"

# One-shot restore request written by Home Assistant.
RESTORE_REQUEST_PATH = "/share/frame_art_restore_request.json"

MYF_RE = re.compile(r"^MY[_-]F(\d+)", re.IGNORECASE)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def write_status(payload: dict) -> None:
    payload["ts"] = time.time()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_options() -> dict:
    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # samsungtvws can return either a top-level dict or a nested "current" dict.
    current_info = current_state.get("current") if isinstance(current_state.get("current"), dict) else {}
    if not current_info:
        current_info = current_state

    return current_info if isinstance(current_info, dict) else {}


def read_restore_request() -> tuple[Optional[str], bool, Optional[bool]]:
    """Return the requested content_id, presence flag, and requested show flag."""
    p = Path(RESTORE_REQUEST_PATH)
    if not p.exists():
        return None, False, None

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        write_status(
            {
                "ok": False,
                "mode": "restore",
                "error": "Malformed restore JSON",
                "exception": repr(e),
            }
        )
        clear_restore_request()
        return None, True, None

    cid = str(payload.get("content_id", "")).strip()
    raw_show = payload.get("show")
    requested_show: Optional[bool]
    if isinstance(raw_show, bool):
        requested_show = raw_show
    elif isinstance(raw_show, str):
        requested_show = raw_show.strip().lower() in {"true", "1", "yes", "on"}
    else:
        requested_show = None

    # Ignore stale restore requests to avoid blocking uploads forever.
    requested_at_raw = payload.get("requested_at")
    if requested_at_raw:
        try:
            req = datetime.fromisoformat(str(requested_at_raw))
            if req.tzinfo is None:
                req = req.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - req.astimezone(timezone.utc)
            if age > timedelta(hours=8):
                clear_restore_request()
                return None, True, requested_show
        except Exception as e:
            write_status(
                {
                    "ok": False,
                    "mode": "restore",
                    "error": "Malformed restore timestamp",
                    "exception": repr(e),
                }
            )
            clear_restore_request()
            return None, True, requested_show

    return (cid or None), True, requested_show


def clear_restore_request() -> None:
    try:
        Path(RESTORE_REQUEST_PATH).unlink()
    except Exception:
        pass


# ------------------------------------------------------------
# Image preparation
# ------------------------------------------------------------

def prepare_for_frame(img_bytes: bytes) -> bytes:
    im = Image.open(BytesIO(img_bytes))

    # Preserve the original mode when possible, but fall back to RGB for unusual
    # formats the TV might not accept.
    if im.mode not in {"RGB", "RGBA", "L"}:
        im = im.convert("RGB")

    w, h = im.size

    target_h = int(round(w * 9 / 16))
    if target_h < h:
        top = (h - target_h) // 2
        im = im.crop((0, top, w, top + target_h))
    elif target_h > h:
        target_w = int(round(h * 16 / 9))
        left = (w - target_w) // 2
        im = im.crop((left, 0, left + target_w, h))

    out = BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    opts = load_options()

    tv_ip = str(opts.get("tv_ip", "")).strip()
    keep_count = int(opts.get("keep_count", 20))
    inbox_dir = str(opts.get("inbox_dir", "/media/frame_ai/inbox"))
    prefix = str(opts.get("filename_prefix", "ai_"))
    select_after = bool(opts.get("select_after_upload", True))

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

    # ------------------------------------------------------------
    # Restore-only path
    # ------------------------------------------------------------
    restore_cid, had_restore_request, requested_show = read_restore_request()
    if restore_cid:
        try:
            # Determine whether Art Mode is currently active. When the TV is in TV mode,
            # get_current() typically returns an empty dict/None, so we fall back to a
            # “not art mode” assumption in that case.
            current_info = get_current_info(art)
            current_id = extract_content_id(current_info)
            is_art_mode = bool(current_info) or current_id is not None
            show_flag = requested_show if requested_show is not None else is_art_mode

            # If we are in Art Mode, show immediately; otherwise queue it silently so we
            # do not force a mode switch.
            art.select_image(restore_cid, show=show_flag)

            verified = False
            if show_flag:
                # Allow multiple checks/retries so slow TV responses don't report false negatives.
                for _ in range(3):
                    time.sleep(1.0)
                    current_info = get_current_info(art)
                    current_id = extract_content_id(current_info)
                    if current_id == restore_cid:
                        verified = True
                        break
                    # Re-issue the selection in case the first attempt didn't latch.
                    art.select_image(restore_cid, show=True)

            write_status(
                {
                    "ok": True if not show_flag else verified,
                    "mode": "restore",
                    "tv_ip": tv_ip,
                    "requested_content_id": restore_cid,
                    "verified": verified,
                    "art_mode": is_art_mode,
                    "requested_show": requested_show,
                }
            )

        except Exception as e:
            write_status(
                {
                    "ok": False,
                    "mode": "restore",
                    "tv_ip": tv_ip,
                    "requested_content_id": restore_cid,
                    "error": repr(e),
                }
            )

        clear_restore_request()
        return
    elif had_restore_request:
        # A restore file existed but was invalid/empty; avoid re-uploading a stale inbox
        # image and report the failure.
        write_status(
            {
                "ok": False,
                "mode": "restore",
                "tv_ip": tv_ip,
                "requested_content_id": restore_cid,
                "error": "Invalid or missing restore content_id",
                "requested_show": requested_show,
            }
        )
        clear_restore_request()
        return

    # ------------------------------------------------------------
    # Upload path
    # ------------------------------------------------------------
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
    processed_bytes = prepare_for_frame(raw_bytes)

    art.upload(processed_bytes, file_type="PNG", matte="none")

    available = art.available()
    myf = extract_myf_ids(available)
    newest = myf[-1][1] if myf else None

    deleted: list[str] = []
    if keep_count > 0 and len(myf) > keep_count:
        to_delete = [cid for (_, cid) in myf[:-keep_count]]
        if to_delete:
            art.delete_list(to_delete)
            deleted = to_delete

    if select_after and newest:
        art.select_image(newest, show=True)

    write_status(
        {
            "ok": True,
            "mode": "upload",
            "tv_ip": tv_ip,
            "uploaded_file": str(img_path),
            "original_file_type": original_type,
            "uploaded_file_type": "PNG",
            "selected_content_id": newest,
            "keep_count": keep_count,
            "myf_count": len(myf),
            "deleted": deleted,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_status({"ok": False, "error": repr(e)})
        raise
