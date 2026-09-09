"""Local-only candidate generation and rendering; never imports the TV uploader."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageChops, ImageFilter
from store import ROOT, db, now, rel, init, guard_spend, safe_file
from fallback import stages_for, advance_after_error

HERE=Path(__file__).resolve().parent
BASE_PROMPT=(HERE/'prompt.txt').read_text().strip()
PROFILES={'original':[(88/255,26,0,16)], 'none':[], 'regular':[(88*88/255/255,26,0,16)],
          'medium':[(.35,55,8,26),(.25,18,0,8)], 'strong':[(.65,80,24,40),(.55,22,0,10)]}

def get_pipeline():
    target=ROOT/'recipe/cover_art.py'
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(HERE.parents[1]/'frame_art_uploader_ai/cover_art.py',target)
        shutil.copyfile(HERE/'prompt.txt',target.parent/'prompt.txt')
    spec=importlib.util.spec_from_file_location('review_cover_art',target)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def composite(background, source, profile, feather_px=24):
    if profile not in PROFILES: raise ValueError('Invalid shadow profile')
    size=1536;x=(background.width-size)//2;y=(background.height-size)//2
    bg=background.convert('RGBA')
    for opacity,blur,spread,offset in PROFILES[profile]:
        alpha=Image.new('L',bg.size,0)
        edge=1 if profile=='original' else 0
        alpha.paste(round(opacity*255),(x-spread,y-spread+offset,x+size+spread+edge,y+size+spread+offset+edge))
        layer=Image.new('RGBA',bg.size,(0,0,0,0));layer.putalpha(alpha.filter(ImageFilter.GaussianBlur(blur)))
        bg=Image.alpha_composite(bg,layer)
    with Image.open(source) as image: cover=image.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    alpha=Image.new('L',(size,size),255)
    for d in range(feather_px):
        t=d/feather_px;v=round(255*t*t*(3-2*t));end=size-d
        for box in [(d,d,end,d+1),(d,end-1,end,end),(d,d,d+1,end),(end-1,d,end,end)]:alpha.paste(v,box)
    final=bg.convert('RGB');final.paste(cover,(x,y),alpha)
    assert ImageChops.difference(final.crop((x+24,y+24,x+1512,y+1512)),cover.crop((24,24,1512,1512))).getbbox() is None
    return final

def render(folder, profile='none', feather_px=None):
    pipeline=get_pipeline()
    background=folder/'background/generated.png';source=folder/'source/cover.png'
    if feather_px not in {None,0,12,24}:raise ValueError('Invalid feather amount')
    name=profile if feather_px is None else f'{profile}-feather-{feather_px}'
    png=folder/'widescreen'/f'{name}.png';jpg=folder/'widescreen-compressed'/f'{name}.jpg'
    if jpg.exists() and png.exists():return jpg
    for path in (png,jpg):path.parent.mkdir(exist_ok=True)
    recipe=json.loads((folder/'recipe.json').read_text())
    with Image.open(background) as image: final=composite(image,source,profile,recipe.get('feather_px',24) if feather_px is None else feather_px)
    temp=png.with_suffix('.tmp');final.save(temp,format='PNG',compress_level=1);temp.replace(png)
    ok,_=pipeline.compress_png_path_to_jpeg_max_bytes(png,jpg,max_bytes=pipeline.JPEG_MAX_BYTES)
    if not ok: raise ValueError('Could not meet the existing TV JPEG size limit')
    return jpg

def cost(usage,model='gpt-image-2.5-flare'):
    if not usage:return None
    parts=usage['input_tokens_details'];output=usage.get('output_tokens_details') or {}
    image_output=output.get('image_tokens',usage['output_tokens'])
    text_output=output.get('text_tokens',0)
    return (parts.get('image_tokens',0)*8+parts['text_tokens']*5+
            image_output*(32 if model=='gpt-image-1.5' else 30)+text_output*10)/1000000

WATER_PROMPT=("Create a seamless, full-bleed underwater background using this water-only reference. "
    "Match its cyan surface light, rippled reflections, fine photographic texture, and transition to deeper blue below. "
    "The reference was assembled from narrow water samples and may have stretched texture: paint natural continuous water, "
    "not stripes or repeated tiles. Keep the same vertical color and brightness distribution so it fits around a centered "
    "square photograph occupying x=461..1075 and y=205..819 on this 1536x1024 canvas. Paint the whole canvas, "
    "including the center. No people, bodies, animals, objects, lettering, logos, borders, shadows, or focal subjects. "
    "The output will be cropped to 16:9 and the original photograph will be composited locally afterward.")

class ProviderError(Exception):
    def __init__(self,response):
        self.status=response.status_code
        self.request_id=response.headers.get('x-request-id')
        super().__init__(f'OpenAI HTTP {self.status}: {response.text[:1000]}')

def post_image(key,files,data):
    # 429 explicitly declined the request; bounded retries do not replay an uncertain image generation.
    for retry in range(3):
        if files is None:
            response=requests.post('https://api.openai.com/v1/images/generations',
                headers={'Authorization':f'Bearer {key}'},json=data,timeout=180)
        else:
            response=requests.post('https://api.openai.com/v1/images/edits',
                headers={'Authorization':f'Bearer {key}'},files=files,data=data,timeout=180)
        if response.status_code!=429 or retry==2:return response
        try:delay=max(1,min(60,float(response.headers.get('retry-after','60'))))
        except ValueError:delay=60
        print(f'Rate limited; waiting {delay:.0f}s before retrying the declined request.',flush=True)
        time.sleep(delay)
    raise AssertionError('unreachable')

def run_worker(batch_id,key,pipeline):
    while True:
        with db() as c:
            c.execute('BEGIN IMMEDIATE')
            batch=c.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
            if not batch or batch['paused']:return
            lifetime_limit=c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]
            if guard_spend(c)+.5>lifetime_limit:
                c.execute("UPDATE batches SET paused=1,pause_reason='Lifetime estimated spend limit reached; Resume cannot reset it' WHERE id=?",(batch_id,))
                print('Batch paused at the lifetime estimated spend limit.',flush=True);return
            spend=guard_spend(c,batch_id,batch['spend_run'])
            if spend+.5>batch['spend_limit']:
                c.execute("UPDATE batches SET paused=1,pause_reason='Estimated spend guard reached' WHERE id=?",(batch_id,))
                print('Batch paused at its estimated spend guard.',flush=True);return
            row=c.execute("SELECT * FROM candidates WHERE batch_id=? AND state='queued' ORDER BY stage DESC,id LIMIT 1",(batch_id,)).fetchone()
            if not row:return
            a=c.execute('SELECT * FROM albums WHERE id=?',(row['album_id'],)).fetchone()
            if a['canonical_id'] or (a['accepted_id'] and not batch['comparison']) or c.execute('SELECT 1 FROM album_selections WHERE album_id=?',(a['id'],)).fetchone():
                c.execute("UPDATE candidates SET state='canceled',error='Album already has an accepted candidate' WHERE id=?",(row['id'],));continue
            c.execute("UPDATE candidates SET state='running',started_at=?,spend_run=? WHERE id=?",
                      (now(),batch['spend_run'],row['id']))
        folder=ROOT/'candidates'/f'{row["id"]:06d}'
        api_succeeded=False
        try:
            stage=stages_for(batch,a)[row['stage']];mode=stage['mode'];model=stage['model']
            if model not in {'gpt-image-2.5-sunburst','gpt-image-2.5-flare','gpt-image-2','gpt-image-1.5'}:raise ValueError('Unsupported model')
            quality=stage.get('quality','auto')
            if quality not in {'auto','low','medium','high'}:raise ValueError('Unsupported quality')
            source=ROOT/a['source_path']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('Source checksum changed')
            frozen_source=json.loads(batch['plan'] or '{}').get('source_checksums',{}).get(a['cache_key'])
            if frozen_source and frozen_source!=a['source_sha256']:raise ValueError('Source changed after this batch was prepared')
            if shutil.disk_usage(ROOT).free<2*1024**3:raise OSError('Less than 2 GiB of free space; batch paused')
            folder.mkdir(parents=True,exist_ok=False)
            (folder/'source').mkdir();(folder/'background').mkdir()
            with Image.open(source) as im:im.convert('RGB').save(folder/'source/cover.png')
            override=None
            if mode=='background':canvas=None
            elif mode=='water':
                override=json.loads(batch['plan'])['reference_overrides'][a['cache_key']]
                reference=ROOT/override['path']
                if override['source_sha256']!=a['source_sha256']:raise ValueError('Water reference source changed')
                canvas=reference.read_bytes()
                if hashlib.sha256(canvas).hexdigest()!=override['sha256']:raise ValueError('Water reference checksum changed')
            elif mode=='repair':
                override=json.loads(batch['plan'])['repair_references'][a['cache_key']]
                if override['source_sha256']!=a['source_sha256']:raise ValueError('Repair source changed')
                canvas=safe_file(override['path']).read_bytes()
                if hashlib.sha256(canvas).hexdigest()!=override['sha256']:raise ValueError('Repair reference changed')
            else:canvas=pipeline.build_reference_canvas_from_album(str(source))
            if canvas is not None:(folder/'reference.png').write_bytes(canvas)
            mask_box=None
            if mode in {'masked','repair'}:
                mask_box=list(pipeline.reference_cover_box(1536,1024))
                mask=Image.new('RGBA',(1536,1024),(0,0,0,0));mask.paste((255,255,255,255),tuple(mask_box))
                mask.save(folder/'mask.png')
            plan=json.loads(batch['plan'] or '{}')
            prompt=plan['background_prompts'][a['cache_key']] if mode=='background' else override['prompt'] if mode=='repair' else WATER_PROMPT if mode=='water' else pipeline.REFERENCE_BACKGROUND_PROMPT if mode=='legacy' else plan.get('masked_prompt',BASE_PROMPT)
            guidance=plan.get('per_album_guidance',{}).get(a['cache_key'])
            with db() as c:
                prior=c.execute("SELECT reasons,notes FROM candidates WHERE album_id=? AND id<? AND decision='rejected' ORDER BY id DESC LIMIT 1",(a['id'],row['id'])).fetchone()
            if guidance is not None:
                if guidance:prompt+='\nReviewer guidance for this attempt: '+guidance
            elif not batch['comparison'] and mode!='water' and prior and json.loads(prior['reasons']):
                prompt+='\nPrevious attempt was rejected for: '+', '.join(json.loads(prior['reasons']))+'. Correct these issues while preserving the original center cover.'
                if prior['notes']:prompt+='\nReviewer guidance: '+prior['notes']
            recipe=dict(model=model,prompt=prompt,size='1536x1024',quality=quality,mode=mode,stage=row['stage'],mask_box=mask_box,
                source_sha256=a['source_sha256'],source_status=a['source_status'],reference_override=override,
                api_input='text_only' if mode=='background' else 'image',
                feather_px=0 if mode=='legacy' else 24,profile='regular',
                cover_art_sha256=hashlib.sha256((ROOT/'recipe/cover_art.py').read_bytes()).hexdigest(),
                rates_per_million=dict(image_input=8,text_input=5,image_output=32 if model=='gpt-image-1.5' else 30))
            (folder/'recipe.json').write_text(json.dumps(recipe,indent=2))
            with db() as c:c.execute('UPDATE candidates SET folder=?,prompt=?,recipe=? WHERE id=?',(rel(folder),prompt,json.dumps(recipe),row['id']))
            print(f'Generating {row["id"]}: {a["artist"]} — {a["title"]} [{model}, {mode}, stage {row["stage"]+1}]',flush=True)
            started=time.perf_counter()
            files=None if mode=='background' else [('image[]',('input.png',canvas,'image/png'))]
            if mode in {'masked','repair'}:files.append(('mask',('mask.png',(folder/'mask.png').read_bytes(),'image/png')))
            response=post_image(key,files,{'model':model,'prompt':prompt,'size':'1536x1024','quality':quality})
            if not response.ok:raise ProviderError(response)
            api_succeeded=True
            payload=response.json();duration=time.perf_counter()-started
            generated=base64.b64decode(payload['data'][0]['b64_json'])
            (folder/'raw.png').write_bytes(generated)
            usage=payload.get('usage');charge=cost(usage,model)
            record=dict(request_id=response.headers.get('x-request-id'),usage=usage,cost=charge,duration=duration,returned_model=payload.get('model'))
            (folder/'response.json').write_text(json.dumps(record,indent=2))
            with db() as c:c.execute('UPDATE candidates SET request_id=?,usage=?,cost=?,duration=? WHERE id=?',
                (record['request_id'],json.dumps(usage),charge,duration,row['id']))
            if payload.get('model') and not (payload['model']==model or payload['model'].startswith(model+'-')):
                raise ValueError('Unexpected returned model')
            background=pipeline.ha_edit_to_frame(generated);background.save(folder/'background/generated.png',compress_level=1)
            render(folder,recipe['profile'])
            with db() as c:
                c.execute("UPDATE candidates SET state='complete',completed_at=? WHERE id=?",(now(),row['id']))
                if charge is None:c.execute("UPDATE batches SET paused=1,pause_reason='Usage missing; inspect billing before continuing' WHERE id=?",(batch_id,))
            print(f'Complete {row["id"]}: {duration:.1f}s, ${charge or 0:.4f}',flush=True)
        except Exception as e:
            provider=isinstance(e,ProviderError)
            uncertain=isinstance(e,(requests.Timeout,requests.ConnectionError)) or api_succeeded or (provider and e.status>=500)
            message=str(e).replace(key,'[redacted]') if key else str(e)
            with db() as c:
                c.execute('BEGIN IMMEDIATE')
                c.execute('UPDATE candidates SET state=?,error=?,completed_at=?,request_id=coalesce(?,request_id) WHERE id=?',
                    ('interrupted' if uncertain else 'error',message[:1500],now(),e.request_id if provider else None,row['id']))
                next_id=None
                if provider and e.status==400:
                    fresh_album=c.execute('SELECT * FROM albums WHERE id=?',(a['id'],)).fetchone()
                    next_id=advance_after_error(c,row,batch,fresh_album)
                if isinstance(e,OSError) or (provider and e.status in {401,402,403,429}):
                    c.execute('UPDATE batches SET paused=1,pause_reason=? WHERE id=?',(message[:300],batch_id))
            if folder.exists():
                try:(folder/'failure.json').write_text(json.dumps({'error':message,'uncertain':uncertain,'next_candidate':next_id},indent=2))
                except OSError:pass  # The database already retains the failure when disk writes fail.
            print(f'Candidate {row["id"]}: {type(e).__name__}; '+(f'fallback queued as {next_id}' if next_id else 'needs inspection'),flush=True)

def run_batch(batch_id):
    init()
    with open(ROOT/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another local generation batch is already running.',flush=True);return
        with db() as c:
            c.execute("UPDATE candidates SET state='interrupted',error='Request outcome uncertain after worker interruption; inspect before retrying' WHERE state='running'")
            batch=c.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
            if not batch or batch['paused']:return
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','root@192.168.1.202',
                'ha apps info ad1c2f89_frame_art_uploader_ai --raw-json'],capture_output=True,check=True,timeout=180)
            key=json.loads(result.stdout)['data']['options']['openai_api_key']
            if not key:raise ValueError('No configured OpenAI API key')
        except Exception:
            with db() as c:c.execute("UPDATE batches SET paused=1,pause_reason='Could not retrieve configured API key over HA SSH; check connection and unlock the SSH agent' WHERE id=?",(batch_id,))
            print('Batch paused: could not load the configured API key over SSH.',flush=True);return
        pipeline=get_pipeline()
        with ThreadPoolExecutor(max_workers=max(1,min(3,batch['workers']))) as pool:
            futures=[pool.submit(run_worker,batch_id,key,pipeline) for _ in range(max(1,min(3,batch['workers']))) ]
            for future in futures:future.result()
        print('Worker finished; inspect batch status for completed, failed, paused, or interrupted entries.',flush=True)

if __name__=='__main__':run_batch(int(sys.argv[1]))
