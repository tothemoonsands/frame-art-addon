"""Import the read-only snapshot and recover covers by recorded Apple collection ID."""
import argparse
import hashlib
import json
import re
import shutil
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageStat
from store import ROOT, db, init, rel, queue_batch

def load(path): return json.loads(path.read_text()) if path.exists() else {}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def import_snapshot():
    init()
    music=ROOT/'snapshot/music'
    inventory={x['path']:x for x in load(ROOT/'snapshot/inventory.json')}
    catalog=load(ROOT/'snapshot/share/frame_art_music_catalog.json').get('entries',{})
    assoc=load(ROOT/'snapshot/share/frame_art_music_associations.json').get('entries',{})
    index=load(music/'index.json').get('entries',{})
    manifest=load(music/'manifest.json').get('entries',{})
    by_id={str(v.get('collection_id') or v.get('collectionId')):v for v in manifest.values() if isinstance(v,dict)}
    by_cache={v.get('cache_key'):v for v in assoc.values() if isinstance(v,dict)}
    with db() as c:
        for filename,entry in catalog.items():
            if entry.get('state')=='deleted': continue
            key=Path(filename).stem.removesuffix('__3840x2160')
            association=by_cache.get(key,{})
            cid=association.get('collection_id') or (int(key) if key.isdigit() else None)
            old=by_id.get(str(cid),{})
            indexed=index.get(key,{})
            text=old.get('text_key') or indexed.get('text_key') or ''
            artist,title=(text.split(' — ',1)+[''])[:2] if ' — ' in text else ('Unknown artist',key)
            artist=association.get('artist') or artist
            title=association.get('album') or title
            current=f'snapshot/music/widescreen-compressed/{filename}' if 'widescreen-compressed/'+filename in inventory else None
            source=next((f'source/{key}{ext}' for ext in ('.jpg','.png','.jpeg') if f'source/{key}{ext}' in inventory),None)
            c.execute('''INSERT INTO albums(cache_key,artist,title,collection_id,current_path,source_path,
              source_status,source_sha256,source_note,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(cache_key) DO NOTHING''',
              (key,artist,title,cid,current,'snapshot/music/'+source if source else None,
               'original' if source else 'missing',inventory[source]['sha256'] if source else None,
               'Original cover from Home Assistant' if source else 'Original cover needs recovery',
               json.dumps(dict(catalog=entry,association=association,index=indexed,legacy=old))))
    print('Imported active catalog entries without changing production keys.',flush=True)

def recover(country='us'):
    folder=ROOT/'recovered/source';folder.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'recovered/lookups';cache.mkdir(exist_ok=True)
    with db() as c:
        rows=[dict(r) for r in c.execute("SELECT * FROM albums WHERE source_status='missing' AND collection_id IS NOT NULL")]
    ids=sorted({r['collection_id'] for r in rows})
    records={}
    for offset in range(0,len(ids),50):
        group=ids[offset:offset+50]
        path=cache/(hashlib.sha256(json.dumps(group).encode()).hexdigest()[:16]+('' if country=='us' else '-'+country)+'.json')
        try:
            if not path.exists():
                response=requests.get('https://itunes.apple.com/lookup',params={'id':','.join(map(str,group)),
                    'entity':'album','country':country},timeout=40)
                response.raise_for_status();payload=response.json()
                path.write_text(json.dumps(payload))
            for item in load(path).get('results',[]):
                if item.get('wrapperType')=='collection' and item.get('collectionId') in group:
                    records[item['collectionId']]=item
            print(f'Looked up {min(offset+50,len(ids))}/{len(ids)} recorded album IDs',flush=True)
        except Exception as e:
            print(f'Lookup group {offset//50+1} needs retry: {type(e).__name__}',flush=True)
    def download(row):
        item=records.get(row['collection_id'])
        if not item:
            return row['id'],None,f'No exact album ID found in the {country.upper()} catalog',None
        url=item.get('artworkUrl100','')
        if urlparse(url).scheme!='https' or not urlparse(url).hostname.endswith('.mzstatic.com'):
            return row['id'],None,'No trusted artwork URL for this album ID',None
        url=re.sub(r'/\d+x\d+bb\.(jpg|png)',r'/3000x3000bb.\1',url)
        path=folder/(row['cache_key']+'.jpg')
        try:
            if not path.exists():
                r=requests.get(url,timeout=45);r.raise_for_status()
                with Image.open(BytesIO(r.content)) as image:
                    image.load()
                    if image.width<300 or image.height<300 or not .65<image.width/image.height<1.5:
                        raise ValueError('Artwork dimensions need manual inspection')
                tmp=path.with_suffix('.tmp');tmp.write_bytes(r.content);tmp.replace(path)
            with Image.open(path) as image: image.verify()
            return row['id'],rel(path),json.dumps({'method':'exact_collection_id','collection_id':row['collection_id'],
                'artist':item.get('artistName'),'album':item.get('collectionName'),'url':url}),digest(path)
        except Exception as e:
            return row['id'],None,'Cover download needs retry: '+type(e).__name__,None
    for count,result in enumerate(ThreadPoolExecutor(max_workers=4).map(download,rows),1):
        album_id,path,note,sha=result
        with db() as c:
            if path:
                c.execute("UPDATE albums SET source_path=?,source_status='matched_id',source_note=?,source_sha256=? WHERE id=?",
                    (path,note,sha,album_id))
            else: c.execute('UPDATE albums SET source_note=? WHERE id=?',(note,album_id))
        if count%25==0 or count==len(rows): print(f'Recovered-cover pass {count}/{len(rows)}',flush=True)
    report()

def verify_snapshot():
    inventory=load(ROOT/'snapshot/inventory.json')
    problems=[]
    for item in inventory:
        p=ROOT/'snapshot/music'/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            problems.append(item['path'])
    result=dict(files=len(inventory),problems=problems,complete=not problems)
    (ROOT/'snapshot/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if problems: raise SystemExit('Snapshot not yet complete or has changed files')

def pilot():
    with db() as c:
        if c.execute('SELECT COUNT(*) FROM albums WHERE pilot=1').fetchone()[0]:
            raise ValueError('Pilot already selected; it will not be charged again automatically')
        rows=[dict(r) for r in c.execute("SELECT * FROM albums WHERE source_status IN ('original','matched_id') AND current_path IS NOT NULL ORDER BY cache_key")]
    buckets={}
    for a in rows:
        source=ROOT/a['source_path'];current=ROOT/a['current_path']
        if not source.exists(): continue
        with Image.open(source) as im:
            small=im.convert('RGB').resize((32,32));stat=ImageStat.Stat(small)
            lum=sum(stat.mean)/3;contrast=sum(stat.stddev)/3
            group=(int(lum//64),int(contrast//32),max(range(3),key=lambda i:stat.mean[i]))
        buckets.setdefault(group,[]).append(a)
    selected=[];artists=set()
    while len(selected)<25 and any(buckets.values()):
        for group in sorted(buckets):
            pool=buckets[group]
            if not pool: continue
            pos=next((i for i,a in enumerate(pool) if a['artist'].casefold() not in artists),0)
            a=pool.pop(pos);selected.append(a);artists.add(a['artist'].casefold())
            if len(selected)==25:break
    if len(selected)!=25: raise ValueError('Need 25 available verified covers before starting the pilot')
    inventory={x['path']:x for x in load(ROOT/'snapshot/inventory.json')}
    for a in selected:
        target=ROOT/a['current_path']
        if not target.exists():
            remote='/media/frame_ai/music/'+str(target.relative_to(ROOT/'snapshot/music'))
            result=subprocess.run(['ssh','-o','BatchMode=yes','root@192.168.1.202','cat -- '+shlex.quote(remote)],capture_output=True,check=True,timeout=60)
            expected=inventory[str(target.relative_to(ROOT/'snapshot/music'))]['sha256']
            if hashlib.sha256(result.stdout).hexdigest()!=expected:raise ValueError('Saved artwork changed during snapshot')
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix('.pilot-copy');temp.write_bytes(result.stdout);temp.replace(target)
    batch=queue_batch([a['id'] for a in selected],'25-album pilot')
    with db() as c:
        for a in selected:c.execute('UPDATE albums SET pilot=1 WHERE id=?',(a['id'],))
    print(json.dumps({'batch_id':batch,'albums':[(a['artist'],a['title']) for a in selected]},indent=2),flush=True)

def report():
    with db() as c:
        result=dict(albums=c.execute('SELECT count(*) FROM albums').fetchone()[0],
            sources={r[0]:r[1] for r in c.execute('SELECT source_status,count(*) FROM albums GROUP BY source_status')},
            missing_saved_images=c.execute('SELECT count(*) FROM albums WHERE current_path IS NULL').fetchone()[0])
    (ROOT/'inventory-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def recover_saved_centers():
    """Prepare explicit source-recovery fallbacks, never silently approve them for generation."""
    with db() as c:rows=[dict(r) for r in c.execute("SELECT * FROM albums WHERE source_status='missing'")]
    inventory={x['path']:x for x in load(ROOT/'snapshot/inventory.json')}
    folder=ROOT/'recovered/source';folder.mkdir(parents=True,exist_ok=True)
    for a in rows:
        if not a['current_path']:continue
        current=ROOT/a['current_path'];remote_rel=str(current.relative_to(ROOT/'snapshot/music'))
        if not current.exists():
            r=subprocess.run(['ssh','-o','BatchMode=yes','root@192.168.1.202','cat -- '+shlex.quote('/media/frame_ai/music/'+remote_rel)],capture_output=True,check=True,timeout=60)
            if hashlib.sha256(r.stdout).hexdigest()!=inventory[remote_rel]['sha256']:raise ValueError('Current image changed')
            current.parent.mkdir(parents=True,exist_ok=True);temp=current.with_suffix('.recovery-copy');temp.write_bytes(r.stdout);temp.replace(current)
        old=json.loads(a['metadata']).get('legacy',{});g=old.get('geometry',{})
        if old.get('preserve_album') and (g.get('album_px_final'),g.get('x0_final'),g.get('y0_final'))==(1536,1152,312):
            with Image.open(current) as image:
                if image.size!=(3840,2160):continue
                image.crop((1152,312,2688,1848)).convert('RGB').save(folder/(a['cache_key']+'.png'))
            path=folder/(a['cache_key']+'.png')
            with db() as c:c.execute("UPDATE albums SET source_path=?,source_status='extracted',source_sha256=?,source_note=? WHERE id=?",
                (rel(path),digest(path),'Cover recovered from the saved JPEG using its recorded original-cover geometry. Inspect before confirming.',a['id']))
        else:
            # A duplicate artist/album alias can use the existing original only if its saved center matches.
            with db() as c:matches=c.execute("SELECT * FROM albums WHERE artist=? AND title=? AND source_status='original' AND id!=?",(a['artist'],a['title'],a['id'])).fetchall()
            if len(matches)!=1:continue
            other=matches[0];source=ROOT/other['source_path']
            if not source.exists():continue
            from PIL import ImageChops
            with Image.open(current) as im, Image.open(source) as src:
                actual=im.crop((1152,312,2688,1848)).convert('RGB').resize((64,64))
                expected=src.convert('RGB').resize((64,64))
                difference=sum(ImageStat.Stat(ImageChops.difference(actual,expected)).mean)/3
            if difference>8:continue
            with db() as c:c.execute("UPDATE albums SET source_path=?,source_status='original',source_sha256=?,source_note=? WHERE id=?",
                (other['source_path'],other['source_sha256'],f'Original cover shared by exact artist/album alias {other["cache_key"]}; saved-center mean difference {difference:.2f}/255.',a['id']))
        print('Recovered saved source:',a['artist'],'—',a['title'],flush=True)
    report()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['import','recover','verify','pilot','report','recover-centers'])
    parser.add_argument('--country',default='us')
    args=parser.parse_args();{'import':import_snapshot,'recover':lambda:recover(args.country),'verify':verify_snapshot,'pilot':pilot,'report':report,'recover-centers':recover_saved_centers}[args.action]()
