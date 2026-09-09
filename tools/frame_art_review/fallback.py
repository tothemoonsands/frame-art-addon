"""Explicit, bounded album-level fallback plans. No network calls."""
import json
from store import db, now

STAGES=[
    {'model':'gpt-image-2.5-flare','mode':'masked'},
    {'model':'gpt-image-2','mode':'masked'},
    {'model':'gpt-image-1.5','mode':'masked'},
    {'model':'gpt-image-2','mode':'legacy'},
    {'model':'gpt-image-1.5','mode':'legacy'},
]

def stages_for(batch,album):
    if batch['mode']!='fallback':return [{'mode':batch['mode'],'model':batch['model']}]
    plan=json.loads(batch['plan'])
    if album['cache_key'] in plan.get('per_album_stages',{}):return plan['per_album_stages'][album['cache_key']]
    if album['cache_key'] in plan.get('reference_overrides',{}):
        return [dict(stage,mode='water') for stage in plan['stages'][:3]]
    return plan['stages']

def advance_after_error(c,candidate,batch,album):
    """Called in the same transaction that records a definitive failed attempt."""
    next_stage=candidate['stage']+1
    if batch['mode']!='fallback' or next_stage>=len(stages_for(batch,album)) or album['accepted_id']:
        return None
    # A resumed worker must never create a second copy of the same fallback request.
    existing=c.execute('SELECT id FROM candidates WHERE batch_id=? AND album_id=? AND stage=?',
                       (batch['id'],album['id'],next_stage)).fetchone()
    if existing:return existing['id']
    return c.execute('INSERT INTO candidates(album_id,batch_id,stage,created_at) VALUES(?,?,?,?)',
                     (album['id'],batch['id'],next_stage,now())).lastrowid

def queue_full_library(plan,spend_limit=50,workers=2):
    """Explicit full-run entry point. Provisional crops remain unconfirmed."""
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute("SELECT 1 FROM candidates WHERE state IN ('queued','running') LIMIT 1").fetchone():
            raise ValueError('An existing queue must finish first')
        if c.execute("SELECT 1 FROM batches WHERE label='Full library · masked with model fallbacks'").fetchone():
            raise ValueError('Full-library run already exists; resume it rather than duplicating charges')
        albums=c.execute('SELECT * FROM albums WHERE accepted_id IS NULL ORDER BY id').fetchall()
        priority={key:i for i,key in enumerate(plan.get('priority_keys',[]))}
        albums.sort(key=lambda a:(priority.get(a['cache_key'],len(priority)),a['id']))
        batch=c.execute("INSERT INTO batches(label,created_at,mode,plan,spend_limit,workers) VALUES(?,?,'fallback',?,?,?)",
            ('Full library · masked with model fallbacks',now(),json.dumps(plan),spend_limit,workers)).lastrowid
        for a in albums:
            available=a['source_path'] and a['source_status'] in {'original','matched_id','confirmed_extract','extracted'}
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at,state,error) VALUES(?,?,?,?,?)',
                (a['id'],batch,now(),'queued' if available else 'error',None if available else 'Source cover unavailable'))
        return batch
