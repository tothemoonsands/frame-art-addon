"""Small, explicit repair pilots. Preserve source assets, reviews and spend controls."""
import hashlib
import json
from io import BytesIO

from PIL import Image

import curation
import pipeline
import store
import workflow


REPAIR_PROMPT = """Repair this existing widescreen composition. The central square at
x=461..1075, y=205..819 on this 1536x1024 image is the original photograph and is
the fixed geometric and color reference. Keep that square at exactly its current
position and size. Edit the surrounding scene to join its four edges continuously.
Use the existing surrounding scene for context, but correct any mismatched geometry,
lighting or duplicated objects at the joins. Continue lines at their exact crossing
points, with the same slope, perspective, scale and texture. One coherent scene.
Do not zoom, move or redraw the central composition. Do not add another copy of
its people, objects or lettering. Remove any shadow, frame or halo around the square;
do not invent a panel or picture inside the image. Preserve the overall palette.
Top and bottom 80 pixels are crop margins. The center will be restored locally;
only the surrounding scene from your output will be visible in the final image.
Specific repair: """


def reference_bytes(background, source):
    """Show the full existing scene, with a hard opaque cover and matching API geometry."""
    with Image.open(background) as im:
        if im.size != (3840,2160):raise ValueError('Repair reference must be a 4K widescreen image')
        scene=im.convert('RGB').resize((1536,864),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(1536,1024))
    canvas.paste(scene,(0,80))
    canvas.paste(scene.crop((0,0,1536,1)).resize((1536,80)),(0,0))
    canvas.paste(scene.crop((0,863,1536,864)).resize((1536,80)),(0,944))
    with Image.open(source) as im:canvas.paste(im.convert('RGB').resize((614,614),Image.Resampling.LANCZOS),(461,205))
    out=BytesIO();canvas.save(out,format='PNG');return out.getvalue()


def queue_pilot(specs, label='Repair pilot · existing scenes'):
    """Explicit selected subset only; never consumes the rest of the draft queue."""
    if not 1<=len(specs)<=6 or len({s['album_id'] for s in specs})!=len(specs):raise ValueError('Choose 1–6 unique repair albums')
    with store.db() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT 1 FROM candidates WHERE state IN ('queued','running')").fetchone():raise ValueError('Finish existing requests first')
        if store.guard_spend(c)+.5>c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]:raise ValueError('Lifetime guard reached')
        if c.execute('SELECT 1 FROM batches WHERE label=?',(label,)).fetchone():raise ValueError('This pilot already exists; inspect or resume it')
        plan={'stages':[{'model':'gpt-image-2.5-sunburst','mode':'repair','quality':'high'}], 'repair_references':{},'per_album_guidance':{}}
        for spec in specs:
            a=workflow.check(c,spec['album_id'],spec['revision'])
            historical=c.execute('SELECT plan FROM batches WHERE id=4').fetchone()
            if historical and a['cache_key'] in json.loads(historical['plan'] or '{}').get('reference_overrides',{}):raise ValueError('Keep the separate water-only reference route for this album')
            if curation.selection(c,a) or workflow.progress(c,a)['status']!='regenerate':raise ValueError('Only unkept regeneration records may enter a repair pilot')
            latest=workflow.latest(c,a['id'])
            if not latest or latest['state']!='complete':raise ValueError('API refusals and uncertain requests need separate handling')
            if not c.execute('SELECT 1 FROM second_pass_queue WHERE album_id=? AND source_sha256=?',(a['id'],a['source_sha256'])).fetchone():raise ValueError('Stage the current source first')
            if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Confirm source first')
            if any(a['id'] in json.loads(g['members']) for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'")):raise ValueError('Resolve duplicates first')
            source=store.safe_file(a['source_path'])
            if hashlib.sha256(source.read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('Source changed')
            info=curation.validate_asset(c,a['id'],spec['asset'])
            # Generated backgrounds have no baked-in shadow. The original is a saved final only.
            if info['kind']=='original':background=store.safe_file(info['path'])
            else:
                g=c.execute('SELECT folder FROM candidates WHERE id=?',(info['candidate_id'],)).fetchone()
                background=store.safe_file(g['folder']+'/background/generated.png')
            raw=reference_bytes(background,source);sha=hashlib.sha256(raw).hexdigest()
            path=store.ROOT/'previews/repair-references'/f'{sha}.png';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(raw)
            plan['repair_references'][a['cache_key']]=dict(path=store.rel(path),sha256=sha,source_sha256=a['source_sha256'],base_asset=spec['asset'],base_sha256=hashlib.sha256(background.read_bytes()).hexdigest(),prompt=REPAIR_PROMPT+spec['instruction'])
            plan['per_album_guidance'][a['cache_key']]='' # Explicit instruction above already incorporates the review.
        batch=c.execute("INSERT INTO batches(label,created_at,mode,plan,workers,spend_limit) VALUES(?,?,'fallback',?,2,50)",(label,store.now(),json.dumps(plan))).lastrowid
        for spec in specs:
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at) VALUES(?,?,?)',(spec['album_id'],batch,store.now()))
            c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(spec['album_id'],))
        curation.event(c,None,'queue targeted repair pilot',None,dict(batch_id=batch,album_ids=[s['album_id'] for s in specs]))
        return batch


def solid_background(album_id,revision,color):
    """Deterministic local background, with the same separate assets and global finish."""
    if len(color)!=3 or any(type(v)!=int or not 0<=v<=255 for v in color):raise ValueError('Use an RGB color')
    with store.db() as c:
        c.execute('BEGIN IMMEDIATE');a=workflow.check(c,album_id,revision)
        if curation.selection(c,a) or workflow.progress(c,a)['status']!='regenerate':raise ValueError('Only an unkept regeneration record may be repaired')
        if (workflow.latest(c,album_id) or {}).get('state') in {'queued','running','interrupted'}:raise ValueError('Resolve existing request first')
        source=store.safe_file(a['source_path'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('Source changed')
        batch=c.execute("INSERT INTO batches(label,created_at,mode) VALUES(?,?,'local')",('Repair pilot · local color match',store.now())).lastrowid
        id=c.execute("INSERT INTO candidates(album_id,batch_id,created_at,state,cost) VALUES(?,?,?,'complete',0)",(album_id,batch,store.now())).lastrowid
        folder=store.ROOT/'candidates'/f'{id:06d}';folder.mkdir(parents=True)
        (folder/'source').mkdir();(folder/'background').mkdir()
        with Image.open(source) as im:im.convert('RGB').save(folder/'source/cover.png')
        Image.new('RGB',(3840,2160),tuple(color)).save(folder/'background/generated.png')
        style=curation.render_style(c)
        recipe=dict(model='local',mode='solid-color',color=color,source_sha256=a['source_sha256'],feather_px=style['feather'],profile=style['shadow'],prompt='Match the black background of the source cover with a solid background; original cover composited locally.',cost=0)
        (folder/'recipe.json').write_text(json.dumps(recipe,indent=2))
        pipeline.render(folder,style['shadow'],style['feather'])
        c.execute('UPDATE candidates SET folder=?,recipe=?,prompt=?,completed_at=? WHERE id=?',(store.rel(folder),json.dumps(recipe),recipe['prompt'],store.now(),id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(album_id,))
        curation.event(c,album_id,'local solid-color repair',None,dict(candidate_id=id,color=color))
        return id


def queue_backgrounds(specs,label):
    """Generate independent scenery/designs from explicit prompts; covers stay local."""
    if not 1<=len(specs)<=50 or len({s['album_id'] for s in specs})!=len(specs):raise ValueError('Choose 1–50 unique albums')
    with store.db() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT 1 FROM candidates WHERE state IN ('queued','running')").fetchone():raise ValueError('Finish existing requests first')
        if c.execute('SELECT 1 FROM batches WHERE label=?',(label,)).fetchone():raise ValueError('This batch already exists; resume it')
        if store.guard_spend(c)+.5>c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]:raise ValueError('Lifetime guard reached')
        plan={'stages':[{'model':'gpt-image-2.5-sunburst','mode':'background','quality':'high'}],'background_prompts':{},'per_album_guidance':{},'source_checksums':{}}
        for spec in specs:
            a=workflow.check(c,spec['album_id'],spec['revision'])
            if curation.selection(c,a) or workflow.progress(c,a)['status']!='regenerate':raise ValueError('Only unkept regeneration records may be queued')
            if (workflow.latest(c,a['id']) or {}).get('state') in {'queued','running','interrupted'}:raise ValueError('Resolve active or uncertain requests first')
            if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Confirm source first')
            if not c.execute('SELECT 1 FROM second_pass_queue WHERE album_id=? AND source_sha256=?',(a['id'],a['source_sha256'])).fetchone():raise ValueError('Stage the current source first')
            if any(a['id'] in json.loads(g['members']) for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'")):raise ValueError('Resolve duplicates first')
            if hashlib.sha256(store.safe_file(a['source_path']).read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('Source changed')
            prompt=spec['prompt']
            if not isinstance(prompt,str) or not 50<=len(prompt)<=5000:raise ValueError('Provide an explicit background design prompt')
            plan['background_prompts'][a['cache_key']]=prompt
            plan['per_album_guidance'][a['cache_key']]=''
            plan['source_checksums'][a['cache_key']]=a['source_sha256']
        batch=c.execute("INSERT INTO batches(label,created_at,mode,plan,workers,spend_limit) VALUES(?,?,'fallback',?,2,50)",(label,store.now(),json.dumps(plan))).lastrowid
        for spec in specs:
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at) VALUES(?,?,?)',(spec['album_id'],batch,store.now()))
            c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(spec['album_id'],))
        curation.event(c,None,'queue independent background designs',None,dict(batch_id=batch,album_ids=[s['album_id'] for s in specs],api_input='text_only'))
        return batch
