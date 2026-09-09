"""Local version choices, duplicate reconciliation, source repair and second-pass drafts."""
import base64
import hashlib
import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path

from PIL import Image
from store import ROOT, db, now, rel, safe_file
from pipeline import PROFILES, render
from fallback import STAGES

RECIPES={
 'sunburst':[{'mode':'masked','model':'gpt-image-2.5-sunburst','quality':'high'}],
 'fallback':STAGES,
 'original-flare':[{'mode':'legacy','model':'gpt-image-2.5-flare'}],
 'original-2':[{'mode':'legacy','model':'gpt-image-2'}],
 'original-1.5':[{'mode':'legacy','model':'gpt-image-1.5'}],
}

def norm(text):
    text=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+',' ',text).strip()

def identity(a):
    artist=norm(re.split(r'\s+(?:&|feat\.?|featuring)\s+',a['artist'],flags=re.I)[0])
    if artist in {'ye','kanye west'}:artist='kanye west'
    title=re.sub(r'\b(?:deluxe|expanded|remaster(?:ed)?|bonus track|anniversary|explicit|clean)\b.*$','',norm(a['title'])).strip()
    title=re.sub(r'\b(?:ep|single|edition|version)\s*$','',title).strip()
    return artist,title

def discover_duplicates():
    with db() as c:
        rows=c.execute('SELECT * FROM albums ORDER BY id').fetchall()
        groups={}
        for a in rows:
            key=identity(a)
            if all(key):groups.setdefault(key,[]).append(a['id'])
        for key,members in groups.items():
            if len(members)<2:continue
            gid=hashlib.sha256(json.dumps(members).encode()).hexdigest()[:16]
            c.execute('INSERT OR IGNORE INTO duplicate_groups(id,members,reason) VALUES(?,?,?)',
                (gid,json.dumps(members),'Matching artist/title after punctuation, artist aliases and edition suffixes; release differences need review.'))

def event(c,album_id,kind,previous,current):
    c.execute('INSERT INTO curation_events(album_id,kind,created_at,previous,current) VALUES(?,?,?,?,?)',
              (album_id,kind,now(),json.dumps(previous),json.dumps(current)))

def root_album(c,album_id):
    a=c.execute('SELECT * FROM albums WHERE id=?',(album_id,)).fetchone()
    if not a:raise ValueError('Album not found')
    if a['canonical_id']:a=c.execute('SELECT * FROM albums WHERE id=?',(a['canonical_id'],)).fetchone()
    return a

def related(c,album_id):
    a=root_album(c,album_id);ids={a['id']}
    for g in c.execute("SELECT * FROM duplicate_groups WHERE status!='separate'"):
        members=json.loads(g['members'])
        if a['id'] in members:ids.update(members)
    ids.update(r[0] for r in c.execute('SELECT id FROM albums WHERE canonical_id=?',(a['id'],)))
    return sorted(ids)

def source_crop(album_id):
    with db() as c:a=c.execute('SELECT * FROM albums WHERE id=?',(album_id,)).fetchone()
    if not a or not a['current_path']:raise ValueError('No original saved image to recover a cover from')
    path=ROOT/'recovered/crops'/f'{a["cache_key"]}-saved-cover.png';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with Image.open(safe_file(a['current_path'])) as im:
            if im.size!=(3840,2160):raise ValueError('Saved artwork has unexpected dimensions')
            im.crop((1152,312,2688,1848)).convert('RGB').save(path)
    return add_source_option(album_id,path,'Cover cropped from original saved artwork; inspect alignment and edition before using')

def add_source_option(album_id,path,origin):
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with db() as c:
        c.execute('INSERT OR IGNORE INTO source_options(album_id,path,sha256,origin) VALUES(?,?,?,?)',(album_id,rel(path),sha,origin))
        return c.execute('SELECT id FROM source_options WHERE album_id=? AND path=?',(album_id,rel(path))).fetchone()[0]

def prepare_sources():
    with db() as c:
        rows=c.execute('SELECT * FROM albums').fetchall()
        wrong={r[0] for r in c.execute("SELECT DISTINCT album_id FROM candidates WHERE reasons LIKE '%Wrong album cover%'")}
    for a in rows:
        if a['source_path'] and (ROOT/a['source_path']).exists():add_source_option(a['id'],ROOT/a['source_path'],a['source_note'] or 'Current generation source')
        if a['id'] in wrong and a['current_path']:source_crop(a['id'])

def assets_for(c,ids):
    assets=[]
    for id in ids:
        a=c.execute('SELECT * FROM albums WHERE id=?',(id,)).fetchone()
        suffix=f' · {a["artist"]} / {a["title"]} [{id}]' if len(ids)>1 else ''
        if a['current_path']:
            assets.append(dict(id=f'original:{id}',kind='original',album_id=id,label='Original saved artwork'+suffix,path=a['current_path'],editable=False,feather=0))
        for g in c.execute("SELECT g.*,b.label batch_label,b.mode batch_mode FROM candidates g JOIN batches b ON b.id=g.batch_id WHERE g.album_id=? ORDER BY g.id",(id,)):
            if g['state']!='complete':continue
            recipe=json.loads(g['recipe'] or '{}')
            label=('First pass' if g['batch_id']==4 else 'Pilot' if g['batch_id']<4 else g['batch_label'])
            assets.append(dict(id=f'candidate:{g["id"]}',kind='candidate',album_id=id,candidate_id=g['id'],batch_id=g['batch_id'],
                label=f'{label} · #{g["id"]} · {recipe.get("model","unknown")} · {g["decision"]}'+suffix,
                model=recipe.get('model'),batch_label=g['batch_label'],mode=recipe.get('mode'),
                editable=True,feather=recipe.get('feather_px',24),decision=g['decision'],revision=g['revision'],source_sha256=recipe.get('source_sha256')))
    return assets

def render_style(c):
    return dict(c.execute('SELECT shadow,feather,revision FROM render_settings WHERE id=1').fetchone())

def save_render_style(shadow,feather,revision):
    if shadow not in PROFILES or type(feather)!=int or feather not in {0,12,24}:raise ValueError('Invalid rendering settings')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');previous=render_style(c)
        if previous['revision']!=revision:raise RuntimeError('Global rendering settings changed; reload before saving')
        c.execute('UPDATE render_settings SET shadow=?,feather=?,revision=revision+1 WHERE id=1',(shadow,feather))
        result=render_style(c)
        event(c,None,'global rendering settings',previous,result)
        return result

def selection(c,a):
    chosen=c.execute('SELECT * FROM album_selections WHERE album_id=?',(a['id'],)).fetchone()
    result=dict(chosen) if chosen else dict(album_id=a['id'],asset=f'candidate:{a["accepted_id"]}',inherited=True) if a['accepted_id'] else None
    if result:
        style=render_style(c) if result['asset'].startswith('candidate:') else {'shadow':'none','feather':0}
        result.update(shadow=style['shadow'],feather=style['feather'])
    return result

def detail(album_id):
    with db() as c:
        a=root_album(c,album_id);ids=related(c,a['id'])
        groups=[dict(g) for g in c.execute('SELECT * FROM duplicate_groups') if a['id'] in json.loads(g['members'])]
        members=[dict(c.execute('SELECT id,artist,title,cache_key,revision,canonical_id FROM albums WHERE id=?',(id,)).fetchone()) for id in ids]
        sources=[dict(r) for id in ids for r in c.execute('SELECT * FROM source_options WHERE album_id=?',(id,))]
        candidates=[dict(r) for id in ids for r in c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC',(id,))]
        candidates.sort(key=lambda row:row['id'],reverse=True)
        queued=c.execute('SELECT * FROM second_pass_queue WHERE album_id=?',(a['id'],)).fetchone()
        return dict(album={k:a[k] for k in a.keys() if k!='metadata'},members=members,groups=groups,assets=assets_for(c,ids),
            selection=selection(c,a),render_style=render_style(c),sources=sources,candidates=candidates,queued=dict(queued) if queued else None,recipes=list(RECIPES))

def validate_asset(c,album_id,asset):
    allowed={a['id']:a for a in assets_for(c,related(c,album_id))}
    if asset not in allowed:raise ValueError('That artwork does not belong to this album or duplicate group')
    return allowed[asset]

def choose(album_id,revision,asset,shadow='regular',feather=0):
    if shadow not in PROFILES or feather not in {0,12,24}:raise ValueError('Invalid preview settings')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=root_album(c,album_id)
        if a['id']!=album_id or a['revision']!=revision:raise RuntimeError('Album changed; reload before choosing artwork')
        info=validate_asset(c,album_id,asset);previous=selection(c,a)
        style=render_style(c);shadow=style['shadow'];feather=style['feather']
        if info['kind']=='original':shadow='none';feather=0
        c.execute('INSERT INTO album_selections(album_id,asset,shadow,feather,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET asset=excluded.asset,shadow=excluded.shadow,feather=excluded.feather,updated_at=excluded.updated_at',
            (album_id,asset,shadow,feather,now()))
        c.execute('UPDATE albums SET accepted_id=?,revision=revision+1 WHERE id=?',(info.get('candidate_id'),album_id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(album_id,))
        event(c,album_id,'choose artwork',previous,dict(asset=asset,shadow=shadow,feather=feather))

def clear_choice(album_id,revision):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=root_album(c,album_id)
        if a['id']!=album_id or a['revision']!=revision:raise RuntimeError('Album changed; reload')
        event(c,album_id,'clear artwork choice',selection(c,a),None)
        c.execute('DELETE FROM album_selections WHERE album_id=?',(album_id,))
        c.execute('UPDATE albums SET accepted_id=NULL,revision=revision+1 WHERE id=?',(album_id,))

def resolve_group(group_id,revision,action,canonical_id=None,asset=None,shadow='regular',feather=0,member_revisions=None):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');g=c.execute('SELECT * FROM duplicate_groups WHERE id=?',(group_id,)).fetchone()
        if not g or g['revision']!=revision:raise RuntimeError('Duplicate group changed; reload')
        ids=json.loads(g['members'])
        if not isinstance(member_revisions,dict) or any(member_revisions.get(str(id))!=c.execute('SELECT revision FROM albums WHERE id=?',(id,)).fetchone()[0] for id in ids):
            raise RuntimeError('An album in this group changed; reload before resolving duplicates')
        if action not in {'merge','separate','undo'}:raise ValueError('Invalid duplicate action')
        if action=='merge':
            if canonical_id not in ids:raise ValueError('Choose a canonical entry from this group')
            info=validate_asset(c,canonical_id,asset)
            if info['album_id'] not in ids:raise ValueError('Choose artwork from this duplicate group')
            if shadow not in PROFILES or feather not in {0,12,24}:raise ValueError('Invalid preview settings')
            style=render_style(c);shadow=style['shadow'];feather=style['feather']
            if info['kind']=='original':shadow='none';feather=0
            for id in ids:
                c.execute('UPDATE albums SET canonical_id=?,revision=revision+1 WHERE id=?',(None if id==canonical_id else canonical_id,id))
                c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))
            c.execute('INSERT INTO album_selections(album_id,asset,shadow,feather,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET asset=excluded.asset,shadow=excluded.shadow,feather=excluded.feather,updated_at=excluded.updated_at',
                (canonical_id,asset,shadow,feather,now()))
            c.execute('UPDATE albums SET accepted_id=? WHERE id=?',(info.get('candidate_id'),canonical_id))
        else:
            for id in ids:c.execute('UPDATE albums SET canonical_id=NULL,revision=revision+1 WHERE id=?',(id,))
        status={'merge':'resolved','separate':'separate','undo':'open'}[action]
        c.execute('UPDATE duplicate_groups SET status=?,canonical_id=?,revision=revision+1 WHERE id=?',(status,canonical_id if action=='merge' else None,group_id))
        event(c,canonical_id,'duplicate '+action,dict(g),dict(group_id=group_id,members=ids,canonical_id=canonical_id,asset=asset))

def select_source(album_id,revision,option_id):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=root_album(c,album_id)
        if a['id']!=album_id or a['revision']!=revision:raise RuntimeError('Album changed; reload')
        option=c.execute('SELECT * FROM source_options WHERE id=?',(option_id,)).fetchone()
        if not option or option['album_id'] not in related(c,album_id):raise ValueError('Source not in this album group')
        if hashlib.sha256(safe_file(option['path']).read_bytes()).hexdigest()!=option['sha256']:raise ValueError('Source file changed')
        previous={k:a[k] for k in ('source_path','source_sha256','source_status')}
        c.execute("UPDATE albums SET source_path=?,source_sha256=?,source_status='user_selected',source_note=?,revision=revision+1 WHERE id=?",(option['path'],option['sha256'],option['origin'],album_id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(album_id,))
        event(c,album_id,'select source cover',previous,dict(option))

def upload_source(album_id,data):
    with db() as c:root_album(c,album_id)
    try:raw=base64.b64decode(data,validate=True)
    except Exception:raise ValueError('Invalid image upload')
    if not 0<len(raw)<=12*1024**2:raise ValueError('Cover must be under 12 MiB')
    with Image.open(BytesIO(raw)) as im:
        if im.width<250 or im.height<250 or im.width*im.height>40_000_000:raise ValueError('Use a cover between 250 pixels and 40 megapixels')
        sha=hashlib.sha256(raw).hexdigest();p=ROOT/'recovered/uploads'/f'{sha}.png';p.parent.mkdir(parents=True,exist_ok=True)
        im.convert('RGB').save(p)
    return add_source_option(album_id,p,'User uploaded cover; choose it after reviewing the preview')

def render_asset(asset,shadow,feather):
    if shadow not in PROFILES or feather not in {0,12,24}:raise ValueError('Invalid preview settings')
    kind,raw_id=asset.split(':');id=int(raw_id)
    with db() as c:
        if kind=='original':
            a=c.execute('SELECT * FROM albums WHERE id=?',(id,)).fetchone()
            if not a:raise ValueError('Unknown album')
            if kind=='original':
                if not a['current_path']:raise ValueError('Original saved artwork unavailable')
                return safe_file(a['current_path'])
        elif kind=='candidate':
            g=c.execute("SELECT folder FROM candidates WHERE id=? AND state='complete'",(id,)).fetchone()
            if not g:raise ValueError('Candidate is not ready')
            return render(ROOT/g['folder'],shadow,feather)
        else:raise ValueError('Unknown artwork type')

def stage_second_pass(album_id,revision,recipe,guidance=''):
    if recipe not in RECIPES:raise ValueError('Unknown second-pass recipe')
    if not isinstance(guidance,str) or len(guidance)>5000:raise ValueError('Guidance must be at most 5000 characters')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=root_album(c,album_id)
        if a['id']!=album_id or a['revision']!=revision:raise RuntimeError('Album changed; reload')
        if selection(c,a):raise ValueError('Clear the kept artwork choice before requesting another generation')
        if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Choose or confirm the source cover first')
        for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'"):
            if album_id in json.loads(g['members']):raise ValueError('Resolve the possible duplicate group or mark its editions separate first')
        latest=c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(album_id,)).fetchone()
        if latest and latest['state'] in {'queued','running','interrupted'}:raise ValueError('Existing or uncertain request needs inspection first')
        if latest and 'Wrong album cover' in json.loads(latest['reasons']) and json.loads(latest['recipe'] or '{}').get('source_sha256')==a['source_sha256']:
            raise ValueError('Select the corrected cover before staging this album')
        c.execute('INSERT INTO second_pass_queue(album_id,recipe,source_sha256,queued_at,guidance) VALUES(?,?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET recipe=excluded.recipe,source_sha256=excluded.source_sha256,queued_at=excluded.queued_at,guidance=excluded.guidance',
            (album_id,recipe,a['source_sha256'],now(),guidance))
        event(c,album_id,'stage second pass',None,{'recipe':recipe,'source_sha256':a['source_sha256'],'guidance':guidance})

def launch_second_pass(limit=25):
    """Launch up to 25 by default; None explicitly launches the entire staged queue."""
    if limit is not None and (type(limit)!=int or not 1<=limit<=25):raise ValueError('Choose 1–25 albums per batch, or None for the full staged queue')
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT 1 FROM candidates WHERE state IN ('queued','running') LIMIT 1").fetchone():raise ValueError('Finish or pause and resolve the existing generation queue first')
        from store import guard_spend
        if guard_spend(c)+.5>c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]:raise ValueError('Lifetime estimated-spend cap reached; your draft is preserved')
        queued=c.execute('SELECT * FROM second_pass_queue ORDER BY queued_at LIMIT ?',(-1 if limit is None else limit,)).fetchall()
        if not queued:raise ValueError('Stage some albums for the second pass first')
        previous=c.execute("SELECT plan FROM batches WHERE id=4").fetchone()
        plan=json.loads(previous['plan']) if previous and previous['plan'] else {'stages':STAGES}
        plan['per_album_stages']={}
        plan['per_album_guidance']={}
        if any(q['recipe']=='sunburst' for q in queued):plan['masked_prompt']=Path(__file__).with_name('continuation-prompt.txt').read_text().strip()
        for q in queued:
            a=c.execute('SELECT * FROM albums WHERE id=?',(q['album_id'],)).fetchone()
            if a['canonical_id'] or selection(c,a) or a['source_sha256']!=q['source_sha256']:raise ValueError('A staged album changed; remove it from the draft queue and stage it again')
            if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Confirm the staged source cover first')
            if any(a['id'] in json.loads(g['members']) for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'")):raise ValueError('A staged album has an unresolved duplicate group')
            if hashlib.sha256(safe_file(a['source_path']).read_bytes()).hexdigest()!=q['source_sha256']:raise ValueError('A staged source file changed')
            latest=c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(a['id'],)).fetchone()
            if latest and latest['state']=='interrupted':raise ValueError('A staged album has an uncertain request')
            stages=RECIPES[q['recipe']]
            # Preserve the reviewed water-only route for Nevermind at every model choice.
            if a['cache_key'] in plan.get('reference_overrides',{}):
                stages=[dict(next(s for s in stages if s['model']==m),mode='water') for m in dict.fromkeys(s['model'] for s in stages)]
            plan['per_album_stages'][a['cache_key']]=stages
            plan['per_album_guidance'][a['cache_key']]=q['guidance']
        batch=c.execute("INSERT INTO batches(label,created_at,mode,plan,workers,spend_limit) VALUES(?,?,'fallback',?,2,50)",('Second pass · selected albums',now(),json.dumps(plan))).lastrowid
        for q in queued:
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at) VALUES(?,?,?)',(q['album_id'],batch,now()))
            c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(q['album_id'],))
        return batch

def restart_with_sunburst(batch_id,album_guidance=None):
    """Replace a paused run's unfinished/unselected albums, preserving every prior result."""
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        old=c.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
        if not old or not old['paused']:raise ValueError('Pause the prior run first')
        if c.execute("SELECT 1 FROM candidates WHERE state='running' OR (state='queued' AND batch_id!=?)",(batch_id,)).fetchone():raise ValueError('Wait for existing requests to finish first')
        from store import guard_spend
        if guard_spend(c)+.5>c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]:raise ValueError('Lifetime estimated-spend cap reached')
        albums=c.execute('SELECT * FROM albums WHERE id IN (SELECT album_id FROM candidates WHERE batch_id=?) ORDER BY id',(batch_id,)).fetchall()
        plan=json.loads(old['plan']);plan['supersedes_batch_id']=batch_id
        plan['stages']=RECIPES['sunburst'];plan['per_album_stages']={}
        plan['masked_prompt']=Path(__file__).with_name('continuation-prompt.txt').read_text().strip()
        plan.setdefault('per_album_guidance',{})
        eligible=[]
        for a in albums:
            if a['canonical_id'] or selection(c,a):continue
            if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Confirm source covers first')
            if hashlib.sha256(safe_file(a['source_path']).read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('A source file changed')
            if any(a['id'] in json.loads(g['members']) for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'")):raise ValueError('Resolve duplicate groups first')
            latest=c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(a['id'],)).fetchone()
            if latest['state']=='interrupted':raise ValueError('Inspect uncertain requests before retrying')
            if 'Wrong album cover' in json.loads(latest['reasons']) and json.loads(latest['recipe'] or '{}').get('source_sha256')==a['source_sha256']:raise ValueError('Correct rejected source covers first')
            override=plan.get('reference_overrides',{}).get(a['cache_key'])
            if override and override['source_sha256']!=a['source_sha256']:raise ValueError('Water-only reference needs updating for the changed source')
            plan['per_album_stages'][a['cache_key']]=[dict(RECIPES['sunburst'][0],mode='water' if override else 'masked')]
            if album_guidance and a['id'] in album_guidance:
                plan['per_album_guidance'][a['cache_key']]='\n'.join(v for v in (plan['per_album_guidance'].get(a['cache_key']),album_guidance[a['id']]) if v)
            eligible.append(a['id'])
        if not eligible:raise ValueError('No albums still need artwork')
        new=c.execute("INSERT INTO batches(label,created_at,paused,mode,model,plan,workers,spend_limit) VALUES(?,?,1,'fallback','gpt-image-2.5-sunburst',?,2,50)",
            ('Second pass · Sunburst',now(),json.dumps(plan))).lastrowid
        c.execute("UPDATE candidates SET state='canceled',error=? WHERE batch_id=? AND state='queued'",(f'Superseded by Sunburst batch {new}',batch_id))
        c.execute('UPDATE batches SET pause_reason=? WHERE id=?',(f'Superseded by Sunburst batch {new}',batch_id))
        for id in eligible:
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at) VALUES(?,?,?)',(id,new,now()))
            c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))
        event(c,None,'restart paused run with Sunburst',{'batch_id':batch_id},{'batch_id':new,'album_ids':eligible,'quality':'high'})
        return new

def reconciliation():
    from workflow import progress
    with db() as c:
        aliases={};entries={};unresolved=[];review_remaining=[]
        for a in c.execute('SELECT * FROM albums ORDER BY id').fetchall():
            root=root_album(c,a['id']);aliases[a['cache_key']]=root['cache_key']
            if a['canonical_id']:continue
            chosen=selection(c,a)
            full_review=progress(c,a)
            if full_review['status']!='confirmed':review_remaining.append(a['id'])
            entries[a['cache_key']]=dict(album_id=a['id'],artist=a['artist'],title=a['title'],collection_id=a['collection_id'],source_path=a['source_path'],source_sha256=a['source_sha256'],source_status=a['source_status'],selection=chosen,
                full_review=full_review,
                aliases=[r['cache_key'] for r in c.execute('SELECT cache_key FROM albums WHERE canonical_id=?',(a['id'],))],
                original_metadata=json.loads(a['metadata']))
            if not chosen:unresolved.append(a['id'])
        groups=[dict(g) for g in c.execute('SELECT * FROM duplicate_groups')]
        return dict(format='frame-art-review-reconciliation-v1',created_at=now(),local_only=True,
            render_style=render_style(c),
            deployment_ready=False,review_complete=not unresolved and not review_remaining and not any(g['status']=='open' for g in groups),
            full_review_remaining=review_remaining,
            original_metadata_by_key={a['cache_key']:json.loads(a['metadata']) for a in c.execute('SELECT cache_key,metadata FROM albums')},
            alias_map=aliases,entries=entries,unselected_album_ids=unresolved,duplicate_groups=groups,
            note='Review plan only. No Home Assistant manifests, catalogs, or TV cache have been changed.')

def enrich_state(result):
    with db() as c:
        groups=[dict(g) for g in c.execute('SELECT * FROM duplicate_groups')]
        chosen={r['album_id']:dict(r) for r in c.execute('SELECT * FROM album_selections')}
        queued={r['album_id']:dict(r) for r in c.execute('SELECT * FROM second_pass_queue')}
        for a in result['albums']:
            own=chosen.get(a['id']);a['chosen_asset']=own['asset'] if own else f'candidate:{a["accepted_id"]}' if a['accepted_id'] else None
            a['draft_recipe']=queued.get(a['id'],{}).get('recipe')
            a['duplicate_group']=next((g['id'] for g in groups if a['id'] in json.loads(g['members']) and g['status']!='separate'),None)
            a['duplicate_status']=next((g['status'] for g in groups if g['id']==a['duplicate_group']),None)
            history=c.execute('SELECT reasons,notes,recipe,state FROM candidates WHERE album_id=? ORDER BY id DESC',(a['id'],)).fetchall()
            a['prefer_original']=any(re.search(r'\boriginal\b|\bprevious(?:ly)?\b|\bold (?:one|version|background|artwork)\b',r['notes'],re.I) for r in history)
            latest=history[0] if history else None
            wrong=latest and 'Wrong album cover' in json.loads(latest['reasons']) and json.loads(latest['recipe'] or '{}').get('source_sha256')==a['source_sha256']
            a['needs_source']=bool(wrong or a['source_status'] in {'missing','extracted'})
        result['duplicate_groups']=groups;result['second_pass_queue']=list(queued.values());result['render_style']=render_style(c)
    return result
