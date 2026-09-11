"""Local final selection of frozen original/updated pairs; never deploys artwork."""
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
from store import ROOT

DATA=ROOT/'reports/final-23-comparison'
LOCK=threading.Lock()
PORT=8767

def state():
    manifest=json.loads((DATA/'manifest.json').read_text())
    p=DATA/'selections.json'
    selection=json.loads(p.read_text()) if p.exists() else {'revision':0,'choices':{}}
    return {**manifest,**selection}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,data,kind='application/json',status=200):
        if not isinstance(data,bytes):data=json.dumps(data).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def valid(self):return self.headers.get('Host') in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}
    def do_GET(self):
        if not self.valid():return self.send({'error':'Invalid host'},status=403)
        try:
            u=urlparse(self.path)
            if u.path=='/':return self.send(Path(__file__).with_suffix('.html').read_bytes(),'text/html; charset=utf-8')
            if u.path=='/api/state':return self.send(state())
            if u.path=='/image':
                q=parse_qs(u.query);a=next(a for a in state()['albums'] if a['id']==int(q['id'][0]));side=q['side'][0]
                if side not in ('original','new'):raise ValueError('Invalid side')
                p=Path(a[side]);return self.send(p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'image/png')
            return self.send({'error':'Not found'},status=404)
        except Exception as e:self.send({'error':str(e)},status=400)
    def do_POST(self):
        if not self.valid() or self.headers.get('Origin') not in (None,f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'):
            return self.send({'error':'Invalid origin'},status=403)
        if self.path!='/api/select':return self.send({'error':'Not found'},status=404)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<10000:raise ValueError('Invalid request')
            body=json.loads(self.rfile.read(size))
            with LOCK:
                s=state()
                if body['revision']!=s['revision']:return self.send({'error':'Another tab changed a selection. Reload to continue.'},status=409)
                a=next(a for a in s['albums'] if a['id']==body['id'])
                choice=body['choice']
                if choice not in ('original','new',None):raise ValueError('Invalid choice')
                if choice is None:s['choices'].pop(str(a['id']),None)
                else:s['choices'][str(a['id'])]={'choice':choice,'path':a[choice],'sha256':a[choice+'_sha256'],'recipe':a['recipe'] if choice=='new' else None,'selected_at':datetime.now(timezone.utc).isoformat()}
                result={'revision':s['revision']+1,'choices':s['choices']}
                temp=DATA/'selections.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(DATA/'selections.json')
            self.send(result)
        except Exception as e:self.send({'error':str(e)},status=400)

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
