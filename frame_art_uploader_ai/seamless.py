"""Reviewed continuation recipe. Mask preservation and final compositing are separate."""
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter

PROMPT = Path(__file__).with_name('continuation-prompt.txt').read_text().strip()
MODELS = ('gpt-image-2.5-flare', 'gpt-image-2.5-sunburst', 'gpt-image-2', 'gpt-image-1.5')
WATER_SOURCE_HASH = 'bdda8b09e4990fed840d00bc02b28a01fc6791b93f8d91684dd644c0e292552f'
WATER_REFERENCE_HASH = '44c5cbecd816177210011a9d3514af1fa112b0bd431f9c4673624d6068e4a61d'
WATER_PROMPT = ('Paint a continuous underwater background from this water-only reference: cyan surface light, '
                'rippled reflections and fine photographic texture transitioning to deeper blue below. '
                'Preserve its vertical color distribution. Replace stretched sample texture with natural water. '
                'Fill the whole canvas. No people, bodies, objects, animals, lettering or focal subjects.')


def png(image):
    out = BytesIO()
    image.save(out, format='PNG', compress_level=1)
    return out.getvalue()


def composite(background, source, shadow=True, feather=0):
    """Match album_cache.py's original-library finish, not the add-on's weaker shadow."""
    if feather != 0 or background.size != (3840, 2160):
        raise ValueError('Original finish requires 3840x2160 and no edge feather')
    base = background.convert('RGBA')
    if shadow:
        layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle((1152, 328, 2688, 1864), fill=(0, 0, 0, 88))
        base = Image.alpha_composite(base, layer.filter(ImageFilter.GaussianBlur(26)))
    with Image.open(source) as im:
        album = im.convert('RGB').resize((1536, 1536), Image.Resampling.LANCZOS)
    base.paste(album, (1152, 312))
    return base.convert('RGB')


class RequestFailure(ValueError):
    def __init__(self, response):
        self.status = response.status_code
        self.request_id = response.headers.get('x-request-id')
        try:
            error = response.json().get('error', {})
        except ValueError:
            error = {}
        self.code = str(error.get('code') or '')
        self.message = str(error.get('message') or '')[:500]
        super().__init__(f'OpenAI HTTP {self.status} code={self.code} request_id={self.request_id}: {self.message}')

    def can_fallback(self):
        # Refused content is never resubmitted to a different model. Transport/server
        # failures may have generated an image, so do not replay those either.
        message = (self.code + ' ' + self.message).lower()
        if any(term in message for term in ('moderation', 'safety', 'policy', 'billing', 'quota')):
            return False
        return self.status in (400, 403, 404, 422) and any(term in message for term in
            ('model', 'verification', 'verified', 'unsupported', 'not supported'))


def request_image(key, model, prompt, canvas, mask, timeout):
    files = [('image[]', ('input.png', canvas, 'image/png'))]
    if mask is not None:
        files.append(('mask', ('mask.png', mask, 'image/png')))
    response = requests.post('https://api.openai.com/v1/images/edits',
        headers={'Authorization': f'Bearer {key}'}, files=files,
        data={'model': model, 'prompt': prompt, 'size': '1536x1024', 'quality': 'high'}, timeout=timeout)
    if not response.ok:
        raise RequestFailure(response)
    payload = response.json()
    returned = payload.get('model') or model
    if returned != model and not returned.startswith(model + '-'):
        raise ValueError('Unexpected image model returned')
    return base64.b64decode(payload['data'][0]['b64_json'], validate=True), response.headers.get('x-request-id'), returned, payload.get('usage')


def generate(source, key, primary, timeout, shadow, hook, cover_art, allow_fallback=True):
    if not key:
        raise ValueError('Missing OpenAI API key')
    if primary not in MODELS:
        raise ValueError(f'Unsupported continuation model: {primary}')
    source = Path(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    water = source_hash == WATER_SOURCE_HASH or source.stem.split('__')[0] == '1440783617' or 'nirvana-nevermind' in source.stem.lower()
    if water:
        reference = Path(__file__).with_name('references') / 'nevermind-water-only.png'
        canvas = reference.read_bytes()
        if hashlib.sha256(canvas).hexdigest() != WATER_REFERENCE_HASH:
            raise ValueError('Water-only reference checksum mismatch')
        mask = None
    else:
        canvas = cover_art.build_reference_canvas_from_album(str(source))
        image = Image.new('RGBA', (1536,1024), (0,0,0,0))
        image.paste((255,255,255,255), cover_art.reference_cover_box(1536,1024))
        mask = png(image)
    stages = [(m, 'water' if water else 'continuation') for m in dict.fromkeys((primary, *MODELS[1:]))]
    if not water:
        stages += [('gpt-image-2', 'legacy'), ('gpt-image-1.5', 'legacy')]
    if not allow_fallback:
        stages = stages[:1]
    attempts = []
    for model, mode in stages:
        prompt = WATER_PROMPT if water else cover_art.REFERENCE_BACKGROUND_PROMPT if mode == 'legacy' else PROMPT
        if hook:
            hook('openai_request_start', {'model': model, 'pipeline': mode, 'mask': mode == 'continuation'})
        try:
            raw, request_id, returned, usage = request_image(key, model, prompt, canvas,
                mask if mode == 'continuation' else None, timeout)
        except RequestFailure as exc:
            attempts.append({'model': model, 'mode': mode, 'code': exc.code, 'request_id': exc.request_id})
            if allow_fallback and exc.can_fallback():
                continue
            raise
        background = cover_art.ha_edit_to_frame(raw)
        final = composite(background, source, shadow)
        recipe = dict(version=1, pipeline=mode, model_requested=primary, model_used=returned,
            prompt=prompt, mask_box=list(cover_art.reference_cover_box(1536,1024)) if mode == 'continuation' else None,
            source_sha256=source_hash, reference_sha256=hashlib.sha256(canvas).hexdigest(),
            shadow='original' if shadow else 'none', feather_px=0, quality='high', size='1536x1024',
            final_size='3840x2160', request_id=request_id, usage=usage, attempts=attempts)
        recipe_path = source.with_suffix('.recipe.json')
        temp = recipe_path.with_suffix('.tmp')
        temp.write_text(json.dumps(recipe, indent=2))
        temp.replace(recipe_path)
        if hook:
            hook('generation_recipe', recipe)
        return png(final), png(background), request_id, returned
    raise ValueError('All supported model/request fallbacks exhausted')
