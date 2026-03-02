import argparse
import base64
import hashlib
import json
import re
import tempfile
import time
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote_plus

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

MUSIC_BASE = Path("/media/frame_ai/music")
SOURCE_DIR = MUSIC_BASE / "source"
WIDESCREEN_DIR = MUSIC_BASE / "widescreen"
BACKGROUND_DIR = MUSIC_BASE / "background"
COMPRESSED_DIR = MUSIC_BASE / "widescreen-compressed"
JPEG_MAX_BYTES = 4 * 1024 * 1024

REFERENCE_BACKGROUND_PROMPT = (
    "Create an original seamless 16:9 full-bleed background inspired by the reference album "
    "cover's color palette, lighting, mood, and visual texture. Keep the result atmospheric "
    "and cohesive for a TV backdrop. Do not include any text, logos, labels, signatures, "
    "watermarks, faces, or copyrighted characters. Do not recreate the exact album cover composition."
)

HA_EDIT_WIDTH = 1536
HA_EDIT_HEIGHT = 1024
HA_EDIT_PASTE_X = 256
HA_EDIT_PASTE_Y = 0
HA_EDIT_SIZE = f"{HA_EDIT_WIDTH}x{HA_EDIT_HEIGHT}"

FRAME_FINAL_WIDTH = 3840
FRAME_FINAL_HEIGHT = 2160

_slug_re = re.compile(r"[^a-z0-9]+")
_artist_split_re = re.compile(r"\s*(?:,|&|\band\b|\bwith\b|\bfeat\.?\b|\bfeaturing\b|\bx\b)\s*", flags=re.IGNORECASE)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _normalize_text(value: str) -> str:
    return _slug_re.sub(" ", value.lower()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in _normalize_text(value).split() if token}


def _artist_variants(artist: str) -> list[str]:
    raw = str(artist or "").strip()
    if not raw:
        return []
    variants: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        normalized = str(candidate or "").strip()
        key = _normalize_text(normalized)
        if not key or key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    add(raw)
    for piece in _artist_split_re.split(raw):
        add(piece)
    return variants


def _canonical_artist_for_cache(artist: str) -> str:
    raw = str(artist or "").strip()
    if not raw:
        return ""
    pieces = _artist_split_re.split(raw)
    for piece in pieces:
        candidate = str(piece or "").strip()
        if _normalize_text(candidate):
            return candidate
    return raw


def _score_itunes_album_result(item: Any, *, wanted_artist: str, wanted_album: str) -> float:
    if not isinstance(item, dict):
        return -1.0
    item_album_raw = str(item.get("collectionName", "")).strip()
    item_artist_raw = str(item.get("artistName", "")).strip()
    if not (item_album_raw or item_artist_raw):
        return -1.0

    wanted_album_norm = _normalize_text(wanted_album)
    wanted_artist_norm = _normalize_text(wanted_artist)
    item_album_norm = _normalize_text(item_album_raw)
    item_artist_norm = _normalize_text(item_artist_raw)

    score = 0.0

    if wanted_album_norm:
        if item_album_norm == wanted_album_norm:
            score += 7.0
        elif wanted_album_norm in item_album_norm or item_album_norm in wanted_album_norm:
            score += 4.0
        score += SequenceMatcher(None, wanted_album_norm, item_album_norm).ratio() * 3.0

    if wanted_artist_norm:
        if item_artist_norm == wanted_artist_norm:
            score += 6.0
        elif wanted_artist_norm in item_artist_norm or item_artist_norm in wanted_artist_norm:
            score += 3.0
        wanted_tokens = _tokens(wanted_artist_norm)
        item_tokens = _tokens(item_artist_norm)
        if wanted_tokens and item_tokens:
            overlap = len(wanted_tokens & item_tokens) / float(len(wanted_tokens | item_tokens))
            score += overlap * 2.0

    if wanted_album_norm and wanted_artist_norm and item_album_norm and item_artist_norm:
        score += SequenceMatcher(None, f"{wanted_artist_norm} {wanted_album_norm}", f"{item_artist_norm} {item_album_norm}").ratio() * 1.5
    return score


def _itunes_search_results(term: str, *, entity: str, limit: int, timeout_s: int) -> list[dict]:
    query = quote_plus(term.strip())
    resp = requests.get(
        f"https://itunes.apple.com/search?term={query}&entity={entity}&limit={limit}",
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json() if resp.text else {}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []
    out: list[dict] = []
    for item in results:
        if isinstance(item, dict):
            out.append(item)
    return out


def _slug(value: str) -> str:
    return _slug_re.sub("-", value.lower()).strip("-")


def normalize_key(collection_id: Optional[int], artist: str, album: str) -> str:
    if collection_id is not None:
        return f"itc_{collection_id}"
    canonical_artist = _canonical_artist_for_cache(artist)
    combined = f"{canonical_artist} {album}".strip()
    combined_norm = _normalize_text(combined) or "unknown"
    slug = _slug(combined) or "unknown"
    short_hash = hashlib.sha1(combined_norm.encode("utf-8")).hexdigest()[:8]
    return f"aa_{slug}_{short_hash}"


def ensure_dirs() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    WIDESCREEN_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)


def itunes_lookup(collection_id: int, timeout_s: int = 10) -> dict:
    resp = requests.get(
        f"https://itunes.apple.com/lookup?id={collection_id}&entity=album",
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json() if resp.text else {}


def itunes_search(artist: str, album: str, timeout_s: int = 10) -> dict:
    artist_variants = _artist_variants(artist) or [str(artist or "").strip()]
    wanted_album = str(album or "").strip()
    collected: list[dict] = []
    seen_ids: set[str] = set()

    for artist_variant in artist_variants:
        term = f"{artist_variant} {wanted_album}".strip()
        for item in _itunes_search_results(term, entity="album", limit=10, timeout_s=timeout_s):
            dedupe_key = str(item.get("collectionId") or item.get("collectionName") or item.get("artistName") or "")
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            collected.append(item)

    if not collected:
        return {}

    best_item: dict[str, Any] = {}
    best_score = -1.0
    for item in collected:
        item_score = max(
            _score_itunes_album_result(item, wanted_artist=variant, wanted_album=wanted_album)
            for variant in artist_variants
        )
        if item_score > best_score:
            best_score = item_score
            best_item = item

    # Reject weak/noisy hits so caller can try alternate resolution paths.
    return best_item if best_score >= 4.0 else {}


def itunes_track_search(artist: str, track: str, timeout_s: int = 10) -> dict:
    artist_variants = _artist_variants(artist) or [str(artist or "").strip()]
    wanted_track = _normalize_text(track)
    if not wanted_track:
        return {}

    collected: list[dict] = []
    seen_ids: set[str] = set()
    for artist_variant in artist_variants:
        term = f"{artist_variant} {track}".strip()
        for item in _itunes_search_results(term, entity="song", limit=10, timeout_s=timeout_s):
            dedupe_key = str(item.get("trackId") or item.get("collectionId") or item.get("trackName") or "")
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            collected.append(item)

    if not collected:
        return {}

    best_item: dict[str, Any] = {}
    best_score = -1.0
    for item in collected:
        item_track = _normalize_text(str(item.get("trackName", "")).strip())
        item_artist = str(item.get("artistName", "")).strip()
        track_score = SequenceMatcher(None, wanted_track, item_track).ratio() * 8.0
        artist_score = max(
            SequenceMatcher(None, _normalize_text(variant), _normalize_text(item_artist)).ratio()
            for variant in artist_variants
        ) * 3.0
        score = track_score + artist_score
        if score > best_score:
            best_score = score
            best_item = item

    return best_item if best_score >= 6.0 else {}


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


def build_reference_canvas_from_album(src_path: str) -> bytes:
    with Image.open(src_path) as src:
        cover = src.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (HA_EDIT_WIDTH, HA_EDIT_HEIGHT), (0, 0, 0))
        canvas.paste(cover, (HA_EDIT_PASTE_X, HA_EDIT_PASTE_Y))

    out = BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()


def build_reference_canvas(src_path: str) -> Path:
    canvas_bytes = build_reference_canvas_from_album(src_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="frame_art_reference_"))
    canvas_path = temp_dir / "reference_canvas.png"
    canvas_path.write_bytes(canvas_bytes)
    return canvas_path


def build_outpaint_canvas_and_mask(src_path: str) -> tuple[Path, Path]:
    """Backward-compatible wrapper that now returns a no-mask reference canvas."""
    canvas_path = build_reference_canvas(src_path)
    return canvas_path, canvas_path


def _validate_openai_multipart_payload(
    files: list[tuple[str, tuple[str, Any, str]]],
    data: dict[str, Any],
) -> None:
    keys = [key for key, _ in files]
    if keys != ["image[]"]:
        raise ValueError(f"OpenAI edits must use only multipart key ['image[]']; got {keys}")
    forbidden = {"image", "image[]"} & set(data.keys())
    if forbidden:
        raise ValueError(f"OpenAI edits form data must not include image fields: {sorted(forbidden)}")


def _request_openai_reference_background_once(
    input_canvas_png: bytes,
    openai_api_key: str,
    openai_model: str,
    prompt: str,
    seed: Optional[int] = None,
    timeout_s: int = 60,
) -> tuple[bytes, Optional[str], Optional[str]]:
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key")

    headers = {"Authorization": f"Bearer {openai_api_key}"}
    files = [("image[]", ("input.png", BytesIO(input_canvas_png), "image/png"))]
    data: dict[str, Any] = {
        "model": openai_model,
        "prompt": prompt,
        "size": HA_EDIT_SIZE,
    }
    if seed is not None:
        data["seed"] = str(seed)
    _validate_openai_multipart_payload(files, data)

    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers=headers,
        files=files,
        data=data,
        timeout=timeout_s,
    )

    request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        body = (response.text or "")[:800]
        raise ValueError(
            f"OpenAI edits failed: {response.status_code} request_id={request_id} body={body}"
        ) from e

    payload = response.json() if response.text else {}
    model_used = payload.get("model") if isinstance(payload, dict) else None
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(
            f"Unexpected OpenAI image response request_id={request_id}: {json.dumps(payload)[:300]}"
        )

    first = items[0] if isinstance(items[0], dict) else {}
    b64_json = first.get("b64_json")
    if not b64_json:
        raise ValueError(
            f"OpenAI response missing b64_json request_id={request_id}: {json.dumps(payload)[:300]}"
        )
    return base64.b64decode(b64_json), request_id, model_used if isinstance(model_used, str) else None


def _request_openai_reference_background(
    input_canvas_png: bytes,
    openai_api_key: str,
    openai_model: str,
    seed: Optional[int] = None,
    timeout_s: int = 60,
) -> tuple[bytes, Optional[str], Optional[str]]:
    return _request_openai_reference_background_once(
        input_canvas_png=input_canvas_png,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        prompt=REFERENCE_BACKGROUND_PROMPT,
        seed=seed,
        timeout_s=timeout_s,
    )


def generate_reference_background(
    input_image_path: str,
    openai_api_key: str,
    openai_model: str,
    seed: Optional[int] = None,
    timeout_s: int = 60,
) -> tuple[bytes, Optional[str], Optional[str]]:
    return _request_openai_reference_background(
        input_canvas_png=Path(input_image_path).read_bytes(),
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        seed=seed,
        timeout_s=timeout_s,
    )


def outpaint_mode_b(
    input_image_path: str,
    input_mask_path: str,
    openai_api_key: str,
    openai_model: str,
    timeout_s: int = 60,
) -> bytes:
    del input_mask_path
    image_bytes, _, _ = generate_reference_background(
        input_image_path,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        timeout_s=timeout_s,
    )
    return image_bytes


def convert_generated_to_background(generated_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(generated_bytes)) as generated:
        im = generated.convert("RGB")
    if im.size != (HA_EDIT_WIDTH, HA_EDIT_HEIGHT):
        raise ValueError(
            f"Generated image size must be {HA_EDIT_WIDTH}x{HA_EDIT_HEIGHT}; got {im.size[0]}x{im.size[1]}"
        )

    target_w = FRAME_FINAL_WIDTH
    upscale_h = int(round(im.height * (target_w / im.width)))
    im = im.resize((target_w, upscale_h), Image.Resampling.LANCZOS)
    if upscale_h < FRAME_FINAL_HEIGHT:
        raise ValueError(f"Upscaled image height too small for center-crop: {upscale_h}")
    top = (upscale_h - FRAME_FINAL_HEIGHT) // 2
    return im.crop((0, top, FRAME_FINAL_WIDTH, top + FRAME_FINAL_HEIGHT))


def ha_edit_to_frame(generated_bytes: bytes) -> Image.Image:
    return convert_generated_to_background(generated_bytes)


def composite_album(
    background: Image.Image,
    source_album_path: Path,
    album_shadow: bool = True,
) -> Image.Image:
    if background.size != (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT):
        raise ValueError(
            f"Background must be {FRAME_FINAL_WIDTH}x{FRAME_FINAL_HEIGHT}; got {background.size[0]}x{background.size[1]}"
        )

    x = (FRAME_FINAL_WIDTH - 1536) // 2
    y = (FRAME_FINAL_HEIGHT - 1536) // 2
    with Image.open(source_album_path) as src:
        album = src.convert("RGBA").resize((1536, 1536), Image.Resampling.LANCZOS)

    final = background.convert("RGBA")

    if album_shadow:
        shadow_layer = Image.new("RGBA", (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT), (0, 0, 0, 0))
        shadow_rect = Image.new("RGBA", (1536, 1536), (0, 0, 0, 88))
        shadow_layer.paste(shadow_rect, (x + 0, y + 16), shadow_rect)
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=26))
        final = Image.alpha_composite(final, shadow_layer)

    album_layer = Image.new("RGBA", (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT), (0, 0, 0, 0))
    album_layer.paste(album, (x, y), album)
    final = Image.alpha_composite(final, album_layer)
    return final.convert("RGB")


def generate_reference_frame_from_album(
    source_album_path: Path,
    openai_api_key: str,
    openai_model: str,
    seed: Optional[int] = None,
    timeout_s: int = 90,
    album_shadow: bool = True,
    step_hook: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> tuple[bytes, bytes, Optional[str], Optional[str]]:
    def emit(stage: str, **fields: Any) -> None:
        if step_hook is not None:
            step_hook(stage, fields)

    t0 = time.perf_counter()
    emit("build_reference_canvas_start", source_path=str(source_album_path))
    reference_canvas_png = build_reference_canvas_from_album(str(source_album_path))
    emit(
        "build_reference_canvas_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        canvas_bytes=len(reference_canvas_png),
    )
    t0 = time.perf_counter()
    emit("openai_request_start", model=openai_model, timeout_s=timeout_s, seed=seed)
    generated_bytes, request_id, model_used = _request_openai_reference_background(
        input_canvas_png=reference_canvas_png,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        seed=seed,
        timeout_s=timeout_s,
    )
    emit(
        "openai_request_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        request_id=request_id,
        model_used=model_used,
        generated_bytes=len(generated_bytes),
    )

    t0 = time.perf_counter()
    emit("background_convert_start")
    background = ha_edit_to_frame(generated_bytes)
    emit(
        "background_convert_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        width=background.width,
        height=background.height,
    )
    t0 = time.perf_counter()
    emit("album_composite_start", album_shadow=album_shadow)
    final = composite_album(background, source_album_path, album_shadow=album_shadow)
    emit(
        "album_composite_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        width=final.width,
        height=final.height,
    )

    t0 = time.perf_counter()
    emit("encode_outputs_start")
    background_out = BytesIO()
    background.save(background_out, format="PNG", optimize=True, compress_level=9)

    final_out = BytesIO()
    final.save(final_out, format="PNG", optimize=True, compress_level=9)
    emit(
        "encode_outputs_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        final_png_bytes=final_out.tell(),
        background_png_bytes=background_out.tell(),
    )

    return final_out.getvalue(), background_out.getvalue(), request_id, model_used


def generate_local_fallback_frame_from_album(
    source_album_path: Path,
    album_shadow: bool = True,
    step_hook: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> tuple[bytes, bytes]:
    def emit(stage: str, **fields: Any) -> None:
        if step_hook is not None:
            step_hook(stage, fields)

    t0 = time.perf_counter()
    emit("fallback_background_start", source_path=str(source_album_path))
    with Image.open(source_album_path) as src:
        source_rgb = src.convert("RGB")
    background = ImageOps.fit(
        source_rgb,
        (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    background = background.filter(ImageFilter.GaussianBlur(radius=62))
    background = ImageEnhance.Color(background).enhance(0.92)
    background = ImageEnhance.Brightness(background).enhance(0.75)

    vignette_mask = Image.new("L", (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT), 0)
    draw = ImageDraw.Draw(vignette_mask)
    max_radius = int((FRAME_FINAL_WIDTH**2 + FRAME_FINAL_HEIGHT**2) ** 0.5 / 2)
    for radius in range(max_radius, 0, -20):
        alpha = int(205 * (1 - radius / max_radius) ** 1.8)
        draw.ellipse(
            (
                FRAME_FINAL_WIDTH // 2 - radius,
                FRAME_FINAL_HEIGHT // 2 - radius,
                FRAME_FINAL_WIDTH // 2 + radius,
                FRAME_FINAL_HEIGHT // 2 + radius,
            ),
            fill=alpha,
        )
    background = Image.composite(
        background,
        Image.new("RGB", (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT), (10, 10, 10)),
        vignette_mask,
    )
    emit(
        "fallback_background_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        width=background.width,
        height=background.height,
    )

    t0 = time.perf_counter()
    emit("fallback_album_composite_start", album_shadow=album_shadow)
    final = composite_album(background, source_album_path, album_shadow=album_shadow)
    emit(
        "fallback_album_composite_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        width=final.width,
        height=final.height,
    )

    t0 = time.perf_counter()
    emit("fallback_encode_outputs_start")
    background_out = BytesIO()
    background.save(background_out, format="PNG", optimize=True, compress_level=9)
    final_out = BytesIO()
    final.save(final_out, format="PNG", optimize=True, compress_level=9)
    emit(
        "fallback_encode_outputs_done",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        final_png_bytes=final_out.tell(),
        background_png_bytes=background_out.tell(),
    )
    return final_out.getvalue(), background_out.getvalue()


def compress_png_path_to_jpeg_max_bytes(
    input_png_path: Path,
    output_jpg_path: Path,
    max_bytes: int = JPEG_MAX_BYTES,
) -> tuple[bool, int]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")

    with Image.open(input_png_path) as img:
        work = img.convert("RGB")

    output_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_jpg_path.with_suffix(".jpg.part")

    def _save(image: Image.Image, quality: int) -> int:
        image.save(tmp_path, format="JPEG", quality=quality, optimize=True, progressive=True)
        return tmp_path.stat().st_size

    selected_size = 0
    selected_ok = False
    for quality in (92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24):
        size = _save(work, quality)
        selected_size = size
        if size <= max_bytes:
            selected_ok = True
            break

    if not selected_ok:
        for scale in (0.95, 0.9, 0.85, 0.8):
            resized = work.resize(
                (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
                Image.Resampling.LANCZOS,
            )
            size = _save(resized, 40)
            selected_size = size
            if size <= max_bytes:
                selected_ok = True
                break

    tmp_path.replace(output_jpg_path)
    return selected_ok, selected_size


def infer_content_id(path: Path) -> Optional[str]:
    stem = path.stem
    if stem.startswith("itc_") and stem[4:].isdigit():
        return stem[4:]
    return None


def unknown_name_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def process_source_file(
    source_path: Path,
    out_dir: Path,
    api_key: str,
    model: str,
    seed: Optional[int],
    save_background_layer: bool,
    album_shadow: bool,
) -> dict[str, Any]:
    content_id = infer_content_id(source_path)
    name_key = content_id if content_id else f"unknown_{unknown_name_hash(source_path)}"

    background_path = out_dir / f"{name_key}__3840x2160__background.png"
    output_path = out_dir / f"{name_key}__3840x2160.png"

    result: dict[str, Any] = {
        "source_path": str(source_path),
        "model_requested": model,
        "model_used": None,
        "prompt_variant": "reference_background_nomask",
        "pipeline": "reference_no_mask",
        "mask_mode": "none",
        "edit_size": "1536x1024",
        "final_size": "3840x2160",
        "background_output_path": str(background_path),
        "output_path": str(output_path),
        "background_bytes": None,
        "output_bytes": None,
        "request_id": None,
        "status": "error",
        "error": None,
    }

    try:
        reference_canvas_png = build_reference_canvas_from_album(str(source_path))
        generated_bytes, request_id, model_used = _request_openai_reference_background(
            input_canvas_png=reference_canvas_png,
            openai_api_key=api_key,
            openai_model=model,
            seed=seed,
            timeout_s=90,
        )
        result["request_id"] = request_id
        result["model_used"] = model_used or model

        background = ha_edit_to_frame(generated_bytes)
        if save_background_layer:
            background.save(background_path, format="PNG", optimize=True, compress_level=9)
            result["background_bytes"] = background_path.stat().st_size

        final = composite_album(background, source_path, album_shadow=album_shadow)
        final.save(output_path, format="PNG", optimize=True, compress_level=9)
        result["output_bytes"] = output_path.stat().st_size

        with Image.open(output_path) as out_im:
            if out_im.size != (FRAME_FINAL_WIDTH, FRAME_FINAL_HEIGHT):
                raise ValueError(f"Final output dimensions invalid: {out_im.size}")

        if save_background_layer and not background_path.exists():
            raise ValueError("Background output file was not created")
        if not output_path.exists():
            raise ValueError("Final composited output file was not created")

        result["status"] = "ok"
    except Exception as e:
        result["error"] = str(e)
    return result


def iter_source_files(source_art_dir: Path) -> list[Path]:
    if not source_art_dir.exists() or not source_art_dir.is_dir():
        return []
    return sorted(
        [p for p in source_art_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def append_manifest(manifest_path: Path, item: dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def run_generate_widescreen(args: argparse.Namespace) -> int:
    source_art_dir = Path(args.source_art_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key or ""
    if not api_key:
        import os

        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing OpenAI API key (use --api-key or OPENAI_API_KEY)")

    files = iter_source_files(source_art_dir)
    manifest_path = out_dir / "manifest.jsonl"

    for source_path in files:
        content_id = infer_content_id(source_path)
        name_key = content_id if content_id else f"unknown_{unknown_name_hash(source_path)}"
        final_path = out_dir / f"{name_key}__3840x2160.png"
        if args.resume and final_path.exists():
            continue

        item = process_source_file(
            source_path=source_path,
            out_dir=out_dir,
            api_key=api_key,
            model=args.model,
            seed=args.seed,
            save_background_layer=args.save_background_layer,
            album_shadow=args.album_shadow,
        )
        append_manifest(manifest_path, item)
        print(json.dumps(item, sort_keys=True))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Album art background generation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-widescreen", help="Generate 3840x2160 widescreen outputs")
    generate.add_argument("--source-art-dir", default=str(SOURCE_DIR), required=False)
    generate.add_argument("--out-dir", default=str(WIDESCREEN_DIR), required=False)
    generate.add_argument("--api-key", default="", required=False)
    generate.add_argument("--model", required=True)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--save-background-layer", dest="save_background_layer", action="store_true")
    generate.add_argument("--no-save-background-layer", dest="save_background_layer", action="store_false")
    generate.set_defaults(save_background_layer=True)
    generate.add_argument("--album-shadow", dest="album_shadow", action="store_true")
    generate.add_argument("--no-album-shadow", dest="album_shadow", action="store_false")
    generate.set_defaults(album_shadow=True)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate-widescreen":
        return run_generate_widescreen(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
