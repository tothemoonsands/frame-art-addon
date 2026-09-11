"""Independent post-deployment review queue; never changes artwork or generation jobs."""
import json

from store import ROOT, db, now
import curation

REASONS = ['Edge / continuation mismatch', 'Shadow mismatch', 'Color / lighting',
           'Wrong cover', 'Scale / placement', 'Unwanted detail', 'Other']


def init():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS manual_reviews (
          album_id INTEGER PRIMARY KEY REFERENCES albums(id),
          status TEXT NOT NULL, notes TEXT NOT NULL, reasons TEXT NOT NULL,
          point TEXT, asset TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS manual_review_events (
          id INTEGER PRIMARY KEY, album_id INTEGER NOT NULL, created_at TEXT NOT NULL,
          previous TEXT, current TEXT NOT NULL);
        ''')


def state():
    release_path = ROOT/'release-original-finish/release.json'
    release = json.loads(release_path.read_text()) if release_path.exists() else {'entries': []}
    released = {e['album_id']: e for e in release['entries']}
    with db() as c:
        rows = c.execute('''SELECT a.id,a.artist,a.title,a.cache_key,a.accepted_id,
          s.asset FROM albums a LEFT JOIN album_selections s ON s.album_id=a.id
          WHERE a.canonical_id IS NULL ORDER BY a.artist COLLATE NOCASE,a.title COLLATE NOCASE,a.id''').fetchall()
        reviews = {r['album_id']: dict(r) for r in c.execute('SELECT * FROM manual_reviews')}
    albums = []
    for row in rows:
        a = dict(row)
        a['asset'] = a['asset'] or (f"candidate:{a['accepted_id']}" if a['accepted_id'] else f"original:{a['id']}")
        e = released.get(a['id'])
        a['release_path'] = e['files']['compressed']['path'] if e and e['asset'] == a['asset'] else None
        a['image_version'] = a['asset']
        r = reviews.get(a['id'], {})
        a.update(status=r.get('status','unreviewed'), notes=r.get('notes',''),
                 reasons=json.loads(r.get('reasons','[]')), points=json.loads(r['point']) if r.get('point') else [],
                 revision=r.get('revision',0), updated_at=r.get('updated_at'), reviewed_asset=r.get('asset'))
        if r and r['asset'] != a['asset']:
            a['status'] = 'unreviewed'
        albums.append(a)
    counts = {s: sum(a['status'] == s for a in albums) for s in ['unreviewed','ok','flagged','later']}
    return dict(albums=albums, counts=counts, total=len(albums), reasons=REASONS)


def artwork(album_id):
    a = next((a for a in state()['albums'] if a['id'] == album_id), None)
    if not a:
        raise ValueError('Album not found')
    if a['release_path']:
        base = (ROOT/'release-original-finish').resolve()
        p = (base/a['release_path']).resolve()
        if not p.is_relative_to(base):
            raise ValueError('Invalid release image')
        return p
    return curation.render_asset(a['asset'], 'original', 0)


def save(album_id, revision, asset, status, notes='', reasons=None, points=None):
    if status not in {'unreviewed','ok','flagged','later'}:
        raise ValueError('Invalid review status')
    if not isinstance(notes,str) or len(notes)>5000:
        raise ValueError('Notes must be at most 5000 characters')
    reasons = reasons or []
    if not isinstance(reasons,list) or any(r not in REASONS for r in reasons):
        raise ValueError('Invalid issue tag')
    points = [] if points is None else points
    if (not isinstance(points,list) or len(points)>30 or any(
            not isinstance(p,list) or len(p)!=2 or
            any(type(x) not in (float,int) or not 0<=x<=1 for x in p) for p in points)):
        raise ValueError('Use up to 30 valid image markers')
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        a = c.execute('''SELECT a.*,s.asset FROM albums a LEFT JOIN album_selections s ON s.album_id=a.id
                         WHERE a.id=? AND a.canonical_id IS NULL''',(album_id,)).fetchone()
        if not a:
            raise ValueError('Album not found')
        current_asset=a['asset'] or (f"candidate:{a['accepted_id']}" if a['accepted_id'] else f"original:{a['id']}")
        if current_asset != asset:
            raise RuntimeError('Artwork changed. Reload before reviewing this version.')
        prior=c.execute('SELECT * FROM manual_reviews WHERE album_id=?',(album_id,)).fetchone()
        if (prior['revision'] if prior else 0) != revision:
            raise RuntimeError('Review changed in another tab. Reload before saving.')
        value=dict(album_id=album_id,status=status,notes=notes,reasons=json.dumps(reasons),
                   point=json.dumps(points),asset=asset,
                   revision=revision+1,updated_at=now())
        c.execute('''INSERT INTO manual_reviews VALUES(:album_id,:status,:notes,:reasons,:point,:asset,:revision,:updated_at)
          ON CONFLICT(album_id) DO UPDATE SET status=excluded.status,notes=excluded.notes,reasons=excluded.reasons,
          point=excluded.point,asset=excluded.asset,revision=excluded.revision,updated_at=excluded.updated_at''',value)
        c.execute('INSERT INTO manual_review_events(album_id,created_at,previous,current) VALUES(?,?,?,?)',
                  (album_id,now(),json.dumps(dict(prior)) if prior else None,json.dumps(value)))
    return {'ok':True,'revision':revision+1}


def export():
    s=state()
    return dict(format='frame-art-manual-fixes-v1',created_at=now(),counts=s['counts'],
                instructions='Preserve original album pixels. Repair background, reapply original cover, show preview before deployment.',
                queue=[a for a in s['albums'] if a['status']=='flagged'])
