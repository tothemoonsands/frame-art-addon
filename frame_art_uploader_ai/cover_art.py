import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

import requests
from PIL import Image

COVER_ART_BASE = Path("/media/frame_ai/cover_art")
SOURCE_DIR = COVER_ART_BASE / "source"
WIDESCREEN_DIR = COVER_ART_BASE / "widescreen"

OUTPAINT_PROMPT = (
    "Remove ALL text, typography, logos, label marks, and signatures from the entire image. "
    "Do not add any new text. Reconstruct removed areas as coherent artwork matching the existing style. "
    "Then outpaint the image into seamless full-bleed 16:9 by extending naturally beyond the left/right edges. "
    "No borders, no mat, no frames, no captions."
)


_slug_re = re.compile(r"[^a-z0-9]+")


def _normalize_text(value: str) -> str:
    return _slug_re.sub(" ", value.lower()).strip()


def _slug(value: str) -> str:
    return _slug_re.sub("-", value.lower()).strip("-")


def normalize_key(collection_id: Optional[int], artist: str, album: str) -> str:
    if collection_id is not None:
        return f"itc_{collection_id}"
    combined = f"{artist} {album}".strip()
    slug = _slug(combined) or "unknown"
    short_hash = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:8]
    return f"aa_{slug}_{short_hash}"


def ensure_dirs() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    WIDESCREEN_DIR.mkdir(parents=True, exist_ok=True)


def itunes_lookup(collection_id: int, timeout_s: int = 10) -> dict:
    resp = requests.get(
        f"https://itunes.apple.com/lookup?id={collection_id}&entity=album",
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def itunes_search(artist: str, album: str, timeout_s: int = 10) -> dict:
    term = quote_plus(f"{artist} {album}".strip())
    resp = requests.get(
        f"https://itunes.apple.com/search?term={term}&entity=album&limit=3",
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json() if resp.text else {}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return {}

    wanted_album = _normalize_text(album)
    wanted_artist = _normalize_text(artist)

    def score(item: Any) -> int:
        if not isinstance(item, dict):
            return -1
        item_album = _normalize_text(str(item.get("collectionName", "")))
        item_artist = _normalize_text(str(item.get("artistName", "")))
        s = 0
        if wanted_album and item_album == wanted_album:
            s += 5
        elif wanted_album and wanted_album in item_album:
            s += 3
        if wanted_artist and item_artist == wanted_artist:
            s += 4
        elif wanted_artist and wanted_artist in item_artist:
            s += 2
        return s

    best = max(results, key=score)
    return best if isinstance(best, dict) else {}


def resolve_artwork_url(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    url = str(result.get("artworkUrl100") or result.get("artworkUrl60") or "").strip()
    if not url:
        return ""
    if "/100x100bb.jpg" in url:
        return url.replace("/100x100bb.jpg", "/3000x3000bb.jpg")
    if "/60x60bb.jpg" in url:
        return url.replace("/60x60bb.jpg", "/3000x3000bb.jpg")
    return re.sub(r"/\d+x\d+bb\.jpg", "/3000x3000bb.jpg", url)


def download_artwork(url: str, dest_path: str, timeout_s: int = 15) -> None:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    Path(dest_path).write_bytes(resp.content)


def build_outpaint_canvas_and_mask(src_path: str) -> tuple[Path, Path]:
    """Build a 1536x1024 outpaint canvas and mask that allows text-band repainting.

    NOTE: If this outpaint algorithm changes, previously cached widescreen images may need
    to be deleted so they can be regenerated with the updated masking behavior.
    """
    canvas_w, canvas_h = 1536, 1024
    center_x, center_y = 256, 0
    center_size = 1024
    top_text_band = int(round(center_size * 0.22))
    bottom_text_band = int(round(center_size * 0.18))
    edge_margin = int(round(center_size * 0.05))

    with Image.open(src_path) as src:
        cover = src.convert("RGB").resize((center_size, center_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        canvas.paste(cover, (center_x, center_y))

    mask = Image.new("L", (canvas_w, canvas_h), 0)
    center_end_x = center_x + center_size

    # Outpaint gutters are fully editable.
    for y in range(canvas_h):
        for x in range(0, center_x):
            mask.putpixel((x, y), 255)
        for x in range(center_end_x, canvas_w):
            mask.putpixel((x, y), 255)

    # Top/bottom bands inside the original square are editable to remove album text.
    for y in range(0, top_text_band):
        for x in range(center_x, center_end_x):
            mask.putpixel((x, y), 255)
    for y in range(center_size - bottom_text_band, center_size):
        for x in range(center_x, center_end_x):
            mask.putpixel((x, y), 255)

    # Small edge margin around the square helps catch edge typography and logos.
    for y in range(0, center_size):
        for i in range(edge_margin):
            left_x = center_x + i
            right_x = center_end_x - 1 - i
            if left_x < center_end_x:
                mask.putpixel((left_x, y), 255)
            if right_x >= center_x:
                mask.putpixel((right_x, y), 255)

    temp_dir = Path(tempfile.mkdtemp(prefix="frame_art_outpaint_"))
    canvas_path = temp_dir / "canvas.png"
    mask_path = temp_dir / "mask.png"
    canvas.save(canvas_path, format="PNG")
    mask.save(mask_path, format="PNG")
    return canvas_path, mask_path


def outpaint_mode_b(
    input_image_path: str,
    input_mask_path: str,
    openai_api_key: str,
    openai_model: str,
    timeout_s: int = 60,
) -> bytes:
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key")
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    with open(input_image_path, "rb") as image_file, open(input_mask_path, "rb") as mask_file:
        files = {
            "image": (Path(input_image_path).name, image_file, "image/png"),
            "mask": (Path(input_mask_path).name, mask_file, "image/png"),
        }
        data = {
            "model": openai_model,
            "prompt": OUTPAINT_PROMPT,
            "size": "1536x1024",
            "response_format": "b64_json",
        }
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            files=files,
            data=data,
            timeout=timeout_s,
        )
    response.raise_for_status()
    payload = response.json() if response.text else {}
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"Unexpected OpenAI image response: {json.dumps(payload)[:300]}")
    first = items[0] if isinstance(items[0], dict) else {}
    b64_json = first.get("b64_json")
    if b64_json:
        import base64

        return base64.b64decode(b64_json)

    url = str(first.get("url", "")).strip()
    if not url:
        raise ValueError("OpenAI response missing image payload")
    img_resp = requests.get(url, timeout=timeout_s)
    img_resp.raise_for_status()
    return img_resp.content
