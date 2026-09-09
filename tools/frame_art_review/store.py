"""Durable local review data. Production catalogs are never modified."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('FRAME_REVIEW_ROOT', str(Path.home() / 'Documents/Frame Art Review'))).resolve()
REASONS = ['Not to scale', 'Not seamless', 'Wrong colors / lighting', 'Distorted imagery',
           'Unwanted text / subjects', 'Wrong album cover', 'Other']

def now(): return datetime.now(timezone.utc).isoformat()

@contextmanager
def db():
    ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ROOT/'review.sqlite3', timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def init():
    with db() as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.executescript('''
        CREATE TABLE IF NOT EXISTS albums (
          id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, artist TEXT NOT NULL,
          title TEXT NOT NULL, collection_id INTEGER, current_path TEXT,
          source_path TEXT, source_status TEXT NOT NULL DEFAULT 'missing', source_sha256 TEXT,
          source_note TEXT, metadata TEXT NOT NULL, pilot INTEGER NOT NULL DEFAULT 0,
          accepted_id INTEGER, revision INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS batches (
          id INTEGER PRIMARY KEY, label TEXT NOT NULL, created_at TEXT NOT NULL,
          paused INTEGER NOT NULL DEFAULT 0, spend_limit REAL NOT NULL DEFAULT 50,
          mode TEXT NOT NULL DEFAULT 'masked', comparison INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS candidates (
          id INTEGER PRIMARY KEY, album_id INTEGER NOT NULL REFERENCES albums(id),
          batch_id INTEGER NOT NULL REFERENCES batches(id), state TEXT NOT NULL DEFAULT 'queued',
          created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, folder TEXT,
          prompt TEXT, recipe TEXT, request_id TEXT, usage TEXT, cost REAL, duration REAL,
          error TEXT, decision TEXT NOT NULL DEFAULT 'unreviewed', reasons TEXT NOT NULL DEFAULT '[]',
          notes TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS candidates_album ON candidates(album_id,id);
        CREATE INDEX IF NOT EXISTS candidates_batch_state ON candidates(batch_id,state);
        CREATE TABLE IF NOT EXISTS review_events (
          id INTEGER PRIMARY KEY, candidate_id INTEGER NOT NULL REFERENCES candidates(id),
          created_at TEXT NOT NULL, previous TEXT NOT NULL, current TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS source_events (
          id INTEGER PRIMARY KEY, album_id INTEGER NOT NULL REFERENCES albums(id),
          created_at TEXT NOT NULL, source_sha256 TEXT NOT NULL, action TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS spend_runs (
          batch_id INTEGER NOT NULL REFERENCES batches(id), run_number INTEGER NOT NULL,
          started_at TEXT NOT NULL, spend_limit REAL NOT NULL,
          PRIMARY KEY(batch_id,run_number));
        CREATE TABLE IF NOT EXISTS spend_settings (
          id INTEGER PRIMARY KEY CHECK(id=1), lifetime_limit REAL NOT NULL);
        INSERT OR IGNORE INTO spend_settings(id,lifetime_limit) VALUES(1,200);
        CREATE TABLE IF NOT EXISTS render_settings (
          id INTEGER PRIMARY KEY CHECK(id=1), shadow TEXT NOT NULL,
          feather INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
        INSERT OR IGNORE INTO render_settings(id,shadow,feather) VALUES(1,'regular',24);
        CREATE TABLE IF NOT EXISTS full_review (
          album_id INTEGER PRIMARY KEY REFERENCES albums(id), status TEXT NOT NULL,
          notes TEXT NOT NULL DEFAULT '', reviewed_revision INTEGER NOT NULL,
          latest_candidate_id INTEGER, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workflow_events (
          id INTEGER PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL,
          previous TEXT NOT NULL, current TEXT NOT NULL, undone_at TEXT);
        CREATE TABLE IF NOT EXISTS duplicate_groups (
          id TEXT PRIMARY KEY, members TEXT NOT NULL, reason TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'open', canonical_id INTEGER, revision INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS album_selections (
          album_id INTEGER PRIMARY KEY REFERENCES albums(id), asset TEXT NOT NULL,
          shadow TEXT NOT NULL, feather INTEGER, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS curation_events (
          id INTEGER PRIMARY KEY, album_id INTEGER, kind TEXT NOT NULL, created_at TEXT NOT NULL,
          previous TEXT NOT NULL, current TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS source_options (
          id INTEGER PRIMARY KEY, album_id INTEGER NOT NULL REFERENCES albums(id),
          path TEXT NOT NULL, sha256 TEXT NOT NULL, origin TEXT NOT NULL,
          UNIQUE(album_id,path));
        CREATE TABLE IF NOT EXISTS second_pass_queue (
          album_id INTEGER PRIMARY KEY REFERENCES albums(id), recipe TEXT NOT NULL,
          source_sha256 TEXT NOT NULL, queued_at TEXT NOT NULL);
        ''')
        columns={r[1] for r in c.execute('PRAGMA table_info(batches)')}
        if 'mode' not in columns:c.execute("ALTER TABLE batches ADD COLUMN mode TEXT NOT NULL DEFAULT 'masked'")
        if 'comparison' not in columns:c.execute('ALTER TABLE batches ADD COLUMN comparison INTEGER NOT NULL DEFAULT 0')
        if 'model' not in columns:c.execute("ALTER TABLE batches ADD COLUMN model TEXT NOT NULL DEFAULT 'gpt-image-2.5-flare'")
        if 'plan' not in columns:c.execute('ALTER TABLE batches ADD COLUMN plan TEXT')
        if 'workers' not in columns:c.execute('ALTER TABLE batches ADD COLUMN workers INTEGER NOT NULL DEFAULT 1')
        if 'pause_reason' not in columns:c.execute('ALTER TABLE batches ADD COLUMN pause_reason TEXT')
        if 'spend_run' not in columns:c.execute('ALTER TABLE batches ADD COLUMN spend_run INTEGER NOT NULL DEFAULT 0')
        candidate_columns={r[1] for r in c.execute('PRAGMA table_info(candidates)')}
        if 'stage' not in candidate_columns:c.execute('ALTER TABLE candidates ADD COLUMN stage INTEGER NOT NULL DEFAULT 0')
        if 'spend_run' not in candidate_columns:c.execute('ALTER TABLE candidates ADD COLUMN spend_run INTEGER NOT NULL DEFAULT 0')
        album_columns={r[1] for r in c.execute('PRAGMA table_info(albums)')}
        if 'canonical_id' not in album_columns:c.execute('ALTER TABLE albums ADD COLUMN canonical_id INTEGER REFERENCES albums(id)')
        queue_columns={r[1] for r in c.execute('PRAGMA table_info(second_pass_queue)')}
        if 'guidance' not in queue_columns:c.execute("ALTER TABLE second_pass_queue ADD COLUMN guidance TEXT NOT NULL DEFAULT ''")

def guard_spend(connection,batch_id=None,run_number=None):
    query="""SELECT coalesce(sum(coalesce(cost,CASE state
        WHEN 'error' THEN .10 WHEN 'interrupted' THEN .5 WHEN 'running' THEN .5 ELSE 0 END)),0)
        FROM candidates"""
    if batch_id is None:return connection.execute(query).fetchone()[0]
    return connection.execute(query+' WHERE batch_id=? AND spend_run=?',(batch_id,run_number)).fetchone()[0]

def resume_batch(batch_id):
    """An explicit resume from pause grants a fresh allowance, without erasing history."""
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        batch=c.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
        if not batch:raise ValueError('Unknown batch')
        lifetime_limit=c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]
        if guard_spend(c)+.5>lifetime_limit:
            raise ValueError(f'Lifetime estimated spend limit of ${lifetime_limit:g} reached; Resume cannot reset it')
        run_number=batch['spend_run']
        # Double-clicks / requests during worker startup must not repeatedly reset the guard.
        if batch['paused'] or run_number==0:
            run_number+=1
            c.execute('INSERT INTO spend_runs(batch_id,run_number,started_at,spend_limit) VALUES(?,?,?,50)',
                      (batch_id,run_number,now()))
        c.execute('UPDATE batches SET paused=0,pause_reason=NULL,spend_limit=50,spend_run=? WHERE id=?',
                  (run_number,batch_id))
        return run_number

def rel(path): return str(Path(path).resolve().relative_to(ROOT))

def safe_file(path):
    resolved=(ROOT/path).resolve()
    if not resolved.is_relative_to(ROOT) or resolved.relative_to(ROOT).parts[0] not in {
        'snapshot','recovered','candidates','previews'}:
        raise ValueError('Invalid image path')
    if resolved.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
        raise ValueError('Only image files are served')
    return resolved

def review(candidate_id, revision, decision, reasons, notes):
    if decision not in {'accepted','rejected','later','unreviewed'}: raise ValueError('Invalid decision')
    if not isinstance(reasons,list) or any(r not in REASONS for r in reasons): raise ValueError('Invalid reason')
    if decision=='rejected' and not reasons: raise ValueError('Choose at least one rejection reason')
    if not isinstance(notes,str) or len(notes)>5000: raise ValueError('Notes must be at most 5000 characters')
    if decision!='rejected': reasons=[]
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        row=c.execute('SELECT * FROM candidates WHERE id=?',(candidate_id,)).fetchone()
        if row is None or row['state']!='complete': raise ValueError('Only completed candidates can be reviewed')
        if row['revision']!=revision: raise RuntimeError('This review changed in another tab. Reload before saving.')
        album=c.execute('SELECT * FROM albums WHERE id=?',(row['album_id'],)).fetchone()
        if album['canonical_id'] or c.execute('SELECT 1 FROM album_selections WHERE album_id=?',(album['id'],)).fetchone():
            raise ValueError('This album has a curated artwork choice or was merged. Use Compare & curate to change its choice first.')
        previous={k:row[k] for k in ('decision','reasons','notes')}
        previous['accepted_id']=album['accepted_id']
        if decision=='accepted' and album['accepted_id'] and album['accepted_id']!=candidate_id:
            old=c.execute('SELECT * FROM candidates WHERE id=?',(album['accepted_id'],)).fetchone()
            old_previous={k:old[k] for k in ('decision','reasons','notes')}
            c.execute("UPDATE candidates SET decision='unreviewed', revision=revision+1 WHERE id=?",(old['id'],))
            c.execute('INSERT INTO review_events(candidate_id,created_at,previous,current) VALUES(?,?,?,?)',
                (old['id'],now(),json.dumps(old_previous),json.dumps({'decision':'unreviewed','replaced_by':candidate_id})))
        c.execute('UPDATE candidates SET decision=?,reasons=?,notes=?,revision=revision+1 WHERE id=?',
                  (decision,json.dumps(reasons),notes,candidate_id))
        c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(row['album_id'],))
        if decision!='accepted' and album['accepted_id']!=candidate_id:
            c.execute('UPDATE albums SET revision=revision+1 WHERE id=?',(row['album_id'],))
        if decision=='accepted':
            c.execute('UPDATE albums SET accepted_id=?,revision=revision+1 WHERE id=?',(candidate_id,row['album_id']))
        elif album['accepted_id']==candidate_id:
            c.execute('UPDATE albums SET accepted_id=NULL,revision=revision+1 WHERE id=?',(row['album_id'],))
        current=dict(decision=decision,reasons=reasons,notes=notes)
        c.execute('INSERT INTO review_events(candidate_id,created_at,previous,current) VALUES(?,?,?,?)',
                  (candidate_id,now(),json.dumps(previous),json.dumps(current)))

def queue_batch(album_ids,label,limit=25):
    ids=list(dict.fromkeys(album_ids))
    if not ids or len(ids)>limit: raise ValueError(f'Choose 1–{limit} albums')
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        eligible=[]
        fallback_plan=None
        for album_id in ids:
            a=c.execute('SELECT * FROM albums WHERE id=?',(album_id,)).fetchone()
            if not a or a['canonical_id'] or a['accepted_id'] or c.execute('SELECT 1 FROM album_selections WHERE album_id=?',(album_id,)).fetchone() or a['source_status'] not in {'original','matched_id','confirmed_extract','user_selected'}:
                raise ValueError('Albums must have verified sources and no accepted candidate')
            latest=c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC LIMIT 1',(album_id,)).fetchone()
            if latest and latest['state'] in {'queued','running','interrupted'}:
                raise ValueError('Album already queued or has an uncertain interrupted request')
            if latest and latest['state']=='complete' and latest['decision']!='rejected':
                raise ValueError('Review the current candidate before requesting another')
            if latest and 'Wrong album cover' in json.loads(latest['reasons']) and a['source_sha256']==json.loads(latest['recipe'] or '{}').get('source_sha256'):
                raise ValueError('Correct the source cover before regenerating')
            next_stage=0
            if latest:
                prior_batch=c.execute('SELECT * FROM batches WHERE id=?',(latest['batch_id'],)).fetchone()
                if prior_batch['mode']=='fallback':
                    from fallback import stages_for
                    next_stage=latest['stage']+1
                    if next_stage>=len(stages_for(prior_batch,a)):
                        raise ValueError('Fallback options exhausted; inspect this album before another request')
                    if fallback_plan and fallback_plan!=prior_batch['plan']:
                        raise ValueError('Choose albums from the same fallback plan')
                    fallback_plan=prior_batch['plan']
            eligible.append((album_id,next_stage))
        batch=c.execute('INSERT INTO batches(label,created_at,mode,plan,workers,spend_limit) VALUES(?,?,?,?,?,50)',
            (label,now(),'fallback' if fallback_plan else 'legacy',fallback_plan,2 if fallback_plan else 1)).lastrowid
        for album_id,stage in eligible:
            c.execute('INSERT INTO candidates(album_id,batch_id,created_at,stage) VALUES(?,?,?,?)',(album_id,batch,now(),stage))
        return batch

def confirm_source(album_id,sha):
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        a=c.execute('SELECT * FROM albums WHERE id=?',(album_id,)).fetchone()
        if not a or a['source_status']!='extracted' or a['source_sha256']!=sha:
            raise ValueError('Source changed or does not need confirmation')
        c.execute("UPDATE albums SET source_status='confirmed_extract' WHERE id=?",(album_id,))
        c.execute('INSERT INTO source_events(album_id,created_at,source_sha256,action) VALUES(?,?,?,?)',
            (album_id,now(),sha,'User confirmed recovered cover'))

def backup():
    folder=ROOT/'backups';folder.mkdir(exist_ok=True)
    path=folder/('review-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.sqlite3')
    with db() as source, sqlite3.connect(path) as dest: source.backup(dest)
    return path
