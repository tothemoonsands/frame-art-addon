"""Fast full-library review. Every action is local, transactional and undoable."""
import hashlib
import json

import curation
from store import db, now, safe_file

FIELDS=('artist','title','source_path','source_sha256','source_status','source_note','accepted_id','canonical_id','revision','collection_id')
TABLES=('album_selections','second_pass_queue','full_review')


def latest(c,id):
    row=c.execute('SELECT id,state FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(id,)).fetchone()
    return dict(row) if row else None


def progress(c,a):
    row=c.execute('SELECT * FROM full_review WHERE album_id=?',(a['id'],)).fetchone()
    result=dict(row) if row else dict(album_id=a['id'],status='pending',notes='')
    newest=latest(c,a['id'])
    if row and (row['reviewed_revision']!=a['revision'] or row['latest_candidate_id']!=(newest or {}).get('id')):
        result['status']='pending'
    return result


def enrich(result):
    with db() as c:
        for a in result['albums']:a['full_review']=progress(c,a)
        result['workflow_last_event']=c.execute('SELECT max(id) FROM workflow_events WHERE undone_at IS NULL').fetchone()[0]
    return result


def detail(id):
    result=curation.detail(id)
    with db() as c:result['full_review']=progress(c,result['album'])
    return result


def check(c,id,revision):
    a=curation.root_album(c,id)
    if a['id']!=id or a['revision']!=revision:raise RuntimeError('This record changed. Reload it before saving.')
    return a


def snapshot(c,ids,groups=False):
    records={}
    for id in ids:
        a=c.execute('SELECT * FROM albums WHERE id=?',(id,)).fetchone()
        r={'album':{k:a[k] for k in FIELDS},'latest':latest(c,id)}
        for table in TABLES:
            row=c.execute('SELECT * FROM '+table+' WHERE album_id=?',(id,)).fetchone()
            r[table]=dict(row) if row else None
        records[str(id)]=r
    result={'records':records}
    if groups:result['groups']=[dict(g) for g in c.execute('SELECT * FROM duplicate_groups ORDER BY id')]
    return result


def mark(c,id,status,notes):
    revision=c.execute('SELECT revision FROM albums WHERE id=?',(id,)).fetchone()[0]
    last=latest(c,id)
    c.execute('INSERT INTO full_review(album_id,status,notes,reviewed_revision,latest_candidate_id,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET status=excluded.status,notes=excluded.notes,reviewed_revision=excluded.reviewed_revision,latest_candidate_id=excluded.latest_candidate_id,updated_at=excluded.updated_at',
              (id,status,notes,revision,(last or {}).get('id'),now()))


def record(c,kind,before):
    after=snapshot(c,[int(id) for id in before['records']],groups='groups' in before)
    return c.execute('INSERT INTO workflow_events(kind,created_at,previous,current) VALUES(?,?,?,?)',(kind,now(),json.dumps(before),json.dumps(after))).lastrowid


def validate_notes(notes):
    if not isinstance(notes,str) or len(notes)>5000:raise ValueError('Notes must be at most 5000 characters')


def keep(c,id,asset,allowed=None):
    info=allowed.get(asset) if allowed is not None else curation.validate_asset(c,id,asset)
    if not info:raise ValueError('Choose artwork belonging to these records')
    style=curation.render_style(c) if info['kind']=='candidate' else {'shadow':'none','feather':0}
    c.execute('INSERT INTO album_selections(album_id,asset,shadow,feather,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(album_id) DO UPDATE SET asset=excluded.asset,shadow=excluded.shadow,feather=excluded.feather,updated_at=excluded.updated_at',
              (id,asset,style['shadow'],style['feather'],now()))
    c.execute('UPDATE albums SET accepted_id=? WHERE id=?',(info.get('candidate_id'),id))
    c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))


def queue(c,a,notes,reasons):
    if a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:raise ValueError('Confirm the source cover first, or flag this album for a fix.')
    if not a['source_path'] or hashlib.sha256(safe_file(a['source_path']).read_bytes()).hexdigest()!=a['source_sha256']:raise ValueError('The source cover is missing or changed; flag this album for a fix.')
    if any(a['id'] in json.loads(g['members']) for g in c.execute("SELECT members FROM duplicate_groups WHERE status='open'")):raise ValueError('Resolve the possible duplicate before queuing another image.')
    row=c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(a['id'],)).fetchone()
    if row and row['state'] in {'queued','running','interrupted'}:raise ValueError('This album already has a queued, active, or uncertain request.')
    if row and 'Wrong album cover' in json.loads(row['reasons']) and json.loads(row['recipe'] or '{}').get('source_sha256')==a['source_sha256']:raise ValueError('Choose the corrected source cover before queuing another image.')
    prior=c.execute("SELECT reasons,notes FROM candidates WHERE album_id=? AND decision='rejected' ORDER BY id DESC LIMIT 1",(a['id'],)).fetchone()
    guidance='\n'.join(v for v in ('; '.join(reasons),notes) if v)
    if not guidance and prior:guidance='\n'.join(v for v in ('; '.join(json.loads(prior['reasons'])),prior['notes']) if v)
    c.execute('DELETE FROM album_selections WHERE album_id=?',(a['id'],))
    c.execute('UPDATE albums SET accepted_id=NULL WHERE id=?',(a['id'],))
    c.execute("INSERT INTO second_pass_queue(album_id,recipe,source_sha256,queued_at,guidance) VALUES(?,'sunburst',?,?,?) ON CONFLICT(album_id) DO UPDATE SET recipe=excluded.recipe,source_sha256=excluded.source_sha256,queued_at=excluded.queued_at,guidance=excluded.guidance",(a['id'],a['source_sha256'],now(),guidance))


def decide(id,revision,action,asset=None,notes='',reasons=None):
    validate_notes(notes)
    from store import REASONS
    reasons=reasons or []
    if not isinstance(reasons,list) or any(r not in REASONS for r in reasons):raise ValueError('Invalid reasons')
    if action not in {'confirm','choose','regenerate','fix','duplicate','later','reopen'}:raise ValueError('Unknown review action')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=check(c,id,revision);before=snapshot(c,[id])
        status=action
        if action in {'confirm','choose'}:
            if action=='confirm':asset=(curation.selection(c,a) or {}).get('asset')
            if not asset:raise ValueError('Choose one of the artwork versions first.')
            keep(c,id,asset);status='confirmed'
        elif action=='regenerate':queue(c,a,notes,reasons)
        elif action=='reopen':status='pending'
        else:c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))
        c.execute('UPDATE albums SET revision=revision+1 WHERE id=?',(id,))
        mark(c,id,status,notes)
        return {'event_id':record(c,action,before),'album_id':id,'status':status}


def edit(id,revision,artist,title,option_id=None,notes='',regenerate=False):
    validate_notes(notes)
    if not all(isinstance(v,str) and 0<len(v.strip())<=300 for v in (artist,title)):raise ValueError('Enter an artist and album title (up to 300 characters each).')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=check(c,id,revision);before=snapshot(c,[id])
        c.execute('UPDATE albums SET artist=?,title=? WHERE id=?',(artist.strip(),title.strip(),id))
        if option_id is not None:
            option=c.execute('SELECT * FROM source_options WHERE id=?',(option_id,)).fetchone()
            if not option or option['album_id'] not in curation.related(c,id):raise ValueError('Source cover does not belong to this record')
            if hashlib.sha256(safe_file(option['path']).read_bytes()).hexdigest()!=option['sha256']:raise ValueError('Source file changed')
            if option['sha256']!=a['source_sha256']:
                c.execute('DELETE FROM album_selections WHERE album_id=?',(id,))
                c.execute('UPDATE albums SET accepted_id=NULL WHERE id=?',(id,))
            c.execute("UPDATE albums SET source_path=?,source_sha256=?,source_status='user_selected',source_note=? WHERE id=?",(option['path'],option['sha256'],option['origin'],id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))
        if regenerate:queue(c,c.execute('SELECT * FROM albums WHERE id=?',(id,)).fetchone(),notes,[])
        c.execute('UPDATE albums SET revision=revision+1 WHERE id=?',(id,))
        mark(c,id,'regenerate' if regenerate else 'pending',notes)
        return {'event_id':record(c,'edit album',before),'album_id':id,'status':'regenerate' if regenerate else 'pending'}


def merge(id,revision,target_id,target_revision,asset,notes=''):
    validate_notes(notes)
    with db() as c:
        c.execute('BEGIN IMMEDIATE');check(c,id,revision);check(c,target_id,target_revision)
        if id==target_id:raise ValueError('Choose another record to merge into')
        ids=sorted(set(curation.related(c,id)+curation.related(c,target_id)))
        before=snapshot(c,ids,groups=True)
        allowed={a['id']:a for a in curation.assets_for(c,ids)}
        if asset not in allowed:raise ValueError('Select artwork from these records')
        if c.execute("SELECT 1 FROM candidates WHERE album_id IN ("+','.join('?' for _ in ids)+") AND state IN ('queued','running','interrupted')",ids).fetchone():raise ValueError('Finish or resolve pending requests for these albums before merging')
        for member in ids:
            c.execute('UPDATE albums SET canonical_id=?,revision=revision+1 WHERE id=?',(None if member==target_id else target_id,member))
            c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(member,))
        for g in c.execute('SELECT * FROM duplicate_groups').fetchall():
            if set(json.loads(g['members'])) & set(ids):c.execute("UPDATE duplicate_groups SET status='resolved',canonical_id=?,revision=revision+1 WHERE id=?",(target_id,g['id']))
        group=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        c.execute("INSERT INTO duplicate_groups(id,members,reason,status,canonical_id) VALUES(?,?,'Manually merged during full review','resolved',?) ON CONFLICT(id) DO UPDATE SET status='resolved',canonical_id=excluded.canonical_id,revision=duplicate_groups.revision+1",(group,json.dumps(ids),target_id))
        keep(c,target_id,asset,allowed)
        mark(c,target_id,'confirmed',notes)
        for member in ids:
            if member!=target_id:mark(c,member,'duplicate',notes)
        return {'event_id':record(c,'merge records',before),'album_id':target_id,'status':'confirmed'}


def correct_catalog(id,revision,option_id):
    """Apply a verified catalog source locally, retaining the kept finished original."""
    with db() as c:
        c.execute('BEGIN IMMEDIATE');a=check(c,id,revision);before=snapshot(c,[id])
        if progress(c,a)['status']!='fix':raise ValueError('Flag this record for a source fix first')
        option=c.execute('SELECT * FROM source_options WHERE id=? AND album_id=?',(option_id,id)).fetchone()
        if not option:raise ValueError('Source option does not belong to this album')
        origin=json.loads(option['origin'])
        if origin.get('method')!='verified_catalog_correction' or type(origin.get('collection_id'))!=int:raise ValueError('Use a verified catalog correction')
        if hashlib.sha256(safe_file(option['path']).read_bytes()).hexdigest()!=option['sha256']:raise ValueError('Source file changed')
        if c.execute("SELECT 1 FROM candidates WHERE album_id=? AND state IN ('queued','running','interrupted')",(id,)).fetchone():raise ValueError('Resolve active or uncertain requests first')
        c.execute("UPDATE albums SET artist=?,title=?,collection_id=?,source_path=?,source_sha256=?,source_status='user_selected',source_note=?,revision=revision+1 WHERE id=?",
                  (origin['artist'],origin['album'],origin['collection_id'],option['path'],option['sha256'],option['origin'],id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(id,))
        mark(c,id,'pending','Corrected catalog association and recovered full-resolution cover. Review the kept original or queue a new generation.')
        return {'event_id':record(c,'correct catalog source',before),'album_id':id}


def undo(event_id):
    with db() as c:
        c.execute('BEGIN IMMEDIATE');e=c.execute('SELECT * FROM workflow_events WHERE id=?',(event_id,)).fetchone()
        if not e or e['undone_at']:raise ValueError('That action is no longer undoable')
        before=json.loads(e['previous']);after=json.loads(e['current']);ids=[int(id) for id in after['records']]
        current=snapshot(c,ids,groups='groups' in after)
        # Old journal entries predate collection-ID corrections; their revision guard still applies.
        for id in ids:
            if 'collection_id' not in after['records'][str(id)]['album']:current['records'][str(id)]['album'].pop('collection_id',None)
        if current!=after:raise RuntimeError('These records changed after that action. Reopen the album to review it safely.')
        for id in ids:
            r=before['records'][str(id)];fields=dict(r['album']);fields['revision']=after['records'][str(id)]['album']['revision']+1
            c.execute('UPDATE albums SET '+','.join(k+'=?' for k in fields)+' WHERE id=?',(*fields.values(),id))
            for table in TABLES:
                c.execute('DELETE FROM '+table+' WHERE album_id=?',(id,))
                row=r[table]
                if row:
                    row=dict(row)
                    if table=='full_review':
                        # Preserve whether the earlier review was already stale.
                        if row['reviewed_revision']==r['album']['revision']:row['reviewed_revision']=fields['revision']
                    c.execute('INSERT INTO '+table+' ('+','.join(row)+') VALUES('+','.join('?' for _ in row)+')',tuple(row.values()))
        if 'groups' in before:
            c.execute('DELETE FROM duplicate_groups')
            for row in before['groups']:c.execute('INSERT INTO duplicate_groups ('+','.join(row)+') VALUES('+','.join('?' for _ in row)+')',tuple(row.values()))
        c.execute('UPDATE workflow_events SET undone_at=? WHERE id=?',(now(),event_id))
        return {'album_id':ids[0]}
