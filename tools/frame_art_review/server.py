"""Loopback-only review app. No endpoints write to Home Assistant or the TV."""
import argparse
import hashlib
import json
import mimetypes
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image
from store import ROOT, REASONS, db, init, review, queue_batch, backup, safe_file, confirm_source, resume_batch, guard_spend
from pipeline import render, PROFILES
from fallback import stages_for
import curation
import workflow

HERE=Path(__file__).resolve().parent
RENDER_LOCK=threading.Lock()
PORT=8766

def state():
    with db() as c:
        rows=[dict(r) for r in c.execute('''SELECT a.*,g.id AS candidate_id,g.state AS generation_state,
          g.decision,g.error,g.batch_id,g.stage,g.recipe FROM albums a LEFT JOIN candidates g ON g.id=coalesce(a.accepted_id,
          (SELECT id FROM candidates WHERE album_id=a.id ORDER BY id DESC LIMIT 1)) ORDER BY a.artist,a.title''')]
        for r in rows:r.pop('metadata',None)
        batches=[dict(r) for r in c.execute('''SELECT b.*,
          (SELECT count(DISTINCT album_id) FROM candidates WHERE batch_id=b.id) AS total,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id) AS attempts,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='complete') AS complete,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='queued') AS queued,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='running') AS running,
          (SELECT count(*) FROM candidates g WHERE batch_id=b.id AND state='error' AND
            g.id=(SELECT max(g2.id) FROM candidates g2 WHERE g2.batch_id=b.id AND g2.album_id=g.album_id)) AS failed,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='error') AS failed_attempts,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='interrupted') AS interrupted,
          (SELECT count(*) FROM candidates WHERE batch_id=b.id AND state='canceled') AS canceled,
          (SELECT coalesce(sum(cost),0) FROM candidates WHERE batch_id=b.id) AS cost
          FROM batches b ORDER BY b.id DESC''')]
        spent=c.execute('SELECT coalesce(sum(cost),0) FROM candidates').fetchone()[0]
        lifetime_guard_spend=guard_spend(c)
        lifetime_spend_limit=c.execute('SELECT lifetime_limit FROM spend_settings WHERE id=1').fetchone()[0]
        by_id={b['id']:b for b in batches}
        for b in batches:b['guard_spend']=guard_spend(c,b['id'],b['spend_run'])
        for r in rows:
            recipe=json.loads(r.pop('recipe') or '{}');b=by_id.get(r['batch_id'])
            stages=stages_for(b,r) if b else []
            stage=stages[r['stage']] if stages and r['stage']<len(stages) else {}
            r['generation_model']=recipe.get('model',stage.get('model'))
            r['generation_mode']=recipe.get('mode',stage.get('mode'))
            r['stages_remaining']=len(stages)-1-r['stage'] if b and b['mode']=='fallback' else None
    migration_path=ROOT/'reports/migration-live.json'
    migration=json.loads(migration_path.read_text()) if migration_path.exists() else None
    return workflow.enrich(curation.enrich_state(dict(migration=migration,albums=rows,batches=batches,reasons=REASONS,profiles=list(PROFILES),spent=spent,
                lifetime_guard_spend=lifetime_guard_spend,lifetime_spend_limit=lifetime_spend_limit)))

def detail(album_id):
    with db() as c:
        a=c.execute('SELECT * FROM albums WHERE id=?',(album_id,)).fetchone()
        if not a:raise ValueError('Album not found')
        candidates=[dict(r) for r in c.execute('SELECT * FROM candidates WHERE album_id=? ORDER BY id DESC',(album_id,))]
        events=[dict(r) for r in c.execute('''SELECT e.* FROM review_events e JOIN candidates g ON e.candidate_id=g.id
            WHERE g.album_id=? ORDER BY e.id DESC''',(album_id,))]
    return dict(album=dict(a),candidates=candidates,events=events)

def start(batch_id):
    resume_batch(batch_id)
    (ROOT/'logs').mkdir(exist_ok=True)
    with open(ROOT/'logs'/f'batch-{batch_id}.log','a') as log:
        worker=subprocess.Popen([sys.executable,str(HERE/'pipeline.py'),str(batch_id)],stdout=log,stderr=log,
                                start_new_session=True)
        threading.Thread(target=worker.wait,daemon=True).start()
        if sys.platform=='darwin':
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(worker.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            threading.Thread(target=awake.wait,daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_bytes(self,data,content_type='application/json',status=200):
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def send_json(self,value,status=200):self.send_bytes(json.dumps(value).encode(),status=status)
    def valid_host(self):return self.headers.get('Host') in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}
    def do_GET(self):
        if not self.valid_host():return self.send_json({'error':'Invalid host'},403)
        try:
            parsed=urlparse(self.path);path=parsed.path;q=parse_qs(parsed.query)
            if path=='/review':return self.send_bytes((HERE/'index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/':return self.send_bytes((HERE/'workbench.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/curation':return self.send_bytes((HERE/'curation.html').read_bytes(),'text/html; charset=utf-8')
            if path.startswith('/api/workflow/'):
                return self.send_json(workflow.detail(int(path.rsplit('/',1)[1])))
            if path=='/api/state':return self.send_json(state())
            if path.startswith('/api/curation/'):
                return self.send_json(curation.detail(int(path.rsplit('/',1)[1])))
            if path=='/api/reconciliation':return self.send_json(curation.reconciliation())
            if path.startswith('/api/album/'):return self.send_json(detail(int(path.rsplit('/',1)[1])))
            if path=='/api/export':
                with db() as c:
                    result={table:[dict(r) for r in c.execute('SELECT * FROM '+table)] for table in ('albums','candidates','batches','review_events','source_events','spend_runs','spend_settings','duplicate_groups','album_selections','curation_events','source_options','second_pass_queue','render_settings','full_review','workflow_events')}
                return self.send_json(result)
            if path.startswith('/files/'):
                file=safe_file(unquote(path[len('/files/'):]))
            elif path=='/preview':
                file=safe_file(q['path'][0]);width=int(q.get('width',['1280'])[0])
                if width not in {240,1280}:raise ValueError('Invalid preview size')
                key=hashlib.sha256((str(file)+str(file.stat().st_mtime_ns)+str(width)).encode()).hexdigest()
                thumb=ROOT/'previews'/(key+'.jpg');thumb.parent.mkdir(exist_ok=True)
                with RENDER_LOCK:
                    if not thumb.exists():
                        with Image.open(file) as im:
                            im=im.convert('RGB');im.thumbnail((width,width));im.save(thumb,quality=88)
                file=thumb
            elif path in {'/asset','/asset-preview'}:
                with RENDER_LOCK:file=curation.render_asset(q['id'][0],q.get('shadow',['regular'])[0],int(q.get('feather',['0'])[0]))
                if path=='/asset-preview':
                    key=hashlib.sha256((str(file)+str(file.stat().st_mtime_ns)+'review-1280').encode()).hexdigest()
                    thumb=ROOT/'previews'/(key+'.jpg');thumb.parent.mkdir(exist_ok=True)
                    with RENDER_LOCK:
                        if not thumb.exists():
                            with Image.open(file) as im:
                                im=im.convert('RGB');im.thumbnail((1280,720));im.save(thumb,quality=90)
                    file=thumb
            elif path.startswith('/render/'):
                _,_,candidate_id,profile=path.split('/')
                if profile not in PROFILES:raise ValueError('Invalid shadow profile')
                with db() as c:
                    row=c.execute("SELECT folder FROM candidates WHERE id=? AND state='complete'",(int(candidate_id),)).fetchone()
                if not row:raise ValueError('Candidate is not ready')
                feather=int(q['feather'][0]) if 'feather' in q else None
                with RENDER_LOCK:file=render(ROOT/row['folder'],profile,feather)
            else:return self.send_json({'error':'Not found'},404)
            self.send_bytes(file.read_bytes(),mimetypes.guess_type(file)[0] or 'application/octet-stream')
        except (ValueError,KeyError,FileNotFoundError) as e:self.send_json({'error':str(e)},400)
        except Exception:self.send_json({'error':'Local operation failed; reload and try again'},500)
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-Frame-Review')!='1':return self.send_json({'error':'Invalid local request'},403)
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}:return self.send_json({'error':'Invalid origin'},403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            maximum=17*1024**2 if self.path=='/api/source-upload' else 20000
            if not 0<length<maximum:raise ValueError('Invalid request size')
            if self.headers.get('Content-Type')!='application/json':raise ValueError('JSON required')
            data=json.loads(self.rfile.read(length))
            if self.path=='/api/workflow-action':
                return self.send_json(workflow.decide(int(data['album_id']),int(data['revision']),data['action'],data.get('asset'),data.get('notes',''),data.get('reasons')))
            if self.path=='/api/workflow-edit':
                return self.send_json(workflow.edit(int(data['album_id']),int(data['revision']),data['artist'],data['title'],data.get('option_id'),data.get('notes',''),data.get('regenerate',False)))
            if self.path=='/api/workflow-merge':
                return self.send_json(workflow.merge(int(data['album_id']),int(data['revision']),int(data['target_id']),int(data['target_revision']),data['asset'],data.get('notes','')))
            if self.path=='/api/workflow-undo':return self.send_json(workflow.undo(int(data['event_id'])))
            if self.path=='/api/render-settings':
                return self.send_json(curation.save_render_style(data['shadow'],int(data['feather']),int(data['revision'])))
            if self.path=='/api/choose':
                curation.choose(int(data['album_id']),int(data['revision']),data['asset'],data.get('shadow','regular'),int(data.get('feather',0)));return self.send_json({'ok':True})
            if self.path=='/api/clear-choice':
                curation.clear_choice(int(data['album_id']),int(data['revision']));return self.send_json({'ok':True})
            if self.path=='/api/duplicates':
                curation.resolve_group(data['group_id'],int(data['revision']),data['action'],data.get('canonical_id'),data.get('asset'),data.get('shadow','regular'),int(data.get('feather',0)),data.get('member_revisions'));return self.send_json({'ok':True})
            if self.path=='/api/source-crop':return self.send_json({'option_id':curation.source_crop(int(data['album_id']))})
            if self.path=='/api/source-upload':return self.send_json({'option_id':curation.upload_source(int(data['album_id']),data['image'])})
            if self.path=='/api/source-select':
                curation.select_source(int(data['album_id']),int(data['revision']),int(data['option_id']));return self.send_json({'ok':True})
            if self.path=='/api/stage-second-pass':
                curation.stage_second_pass(int(data['album_id']),int(data['revision']),data['recipe'],data.get('guidance',''));return self.send_json({'ok':True})
            if self.path=='/api/unstage-second-pass':
                with db() as c:c.execute('DELETE FROM second_pass_queue WHERE album_id=?',(int(data['album_id']),))
                return self.send_json({'ok':True})
            if self.path=='/api/launch-second-pass':
                batch_id=curation.launch_second_pass();backup();start(batch_id);return self.send_json({'batch_id':batch_id})
            if self.path=='/api/confirm-source':
                confirm_source(int(data['album_id']),data['sha256']);return self.send_json({'ok':True})
            if self.path=='/api/review':
                review(int(data['candidate_id']),int(data['revision']),data['decision'],data.get('reasons',[]),data.get('notes',''))
                return self.send_json({'ok':True})
            if self.path=='/api/batch':
                with db() as c:
                    if c.execute("SELECT count(*) FROM candidates WHERE state IN ('queued','running')").fetchone()[0]:
                        raise ValueError('Finish the existing queue before creating a new batch')
                ids=data['album_ids']
                if not isinstance(ids,list) or any(type(x)!=int for x in ids):raise ValueError('Invalid album list')
                batch_id=queue_batch(ids,'Reviewer regeneration batch');backup();start(batch_id)
                return self.send_json({'batch_id':batch_id})
            if self.path=='/api/pause':
                with db() as c:c.execute("UPDATE batches SET paused=1,pause_reason='Paused by reviewer' WHERE id=?",(int(data['batch_id']),))
                return self.send_json({'ok':True})
            if self.path=='/api/resume':start(int(data['batch_id']));return self.send_json({'ok':True})
            if self.path=='/api/backup':backup();return self.send_json({'ok':True})
            self.send_json({'error':'Not found'},404)
        except RuntimeError as e:self.send_json({'error':str(e)},409)
        except (ValueError,KeyError,TypeError) as e:self.send_json({'error':str(e)},400)
        except Exception:self.send_json({'error':'Could not save; your previous decisions are unchanged'},500)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766)
    PORT=parser.parse_args().port;init()
    print(f'Local review: http://127.0.0.1:{PORT}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
