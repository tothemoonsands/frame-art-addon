import base64
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests
from PIL import Image
import store
import pipeline
from fallback import STAGES, queue_full_library

class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.patches=[mock.patch.object(store,'ROOT',self.root),mock.patch.object(pipeline,'ROOT',self.root)]
        for patch in self.patches:patch.start()
        store.init();(self.root/'recipe').mkdir();(self.root/'recipe/cover_art.py').write_text('frozen test compositor')
        self.source=self.root/'cover.png';Image.new('RGB',(20,20),'red').save(self.source)
        self.sha=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.add_album('a')
        raw=BytesIO();Image.new('RGB',(24,16),'blue').save(raw,format='PNG')
        self.payload={'data':[{'b64_json':base64.b64encode(raw.getvalue()).decode()}],
                      'usage':{'input_tokens_details':{'image_tokens':100,'text_tokens':10},'output_tokens':100}}
        self.module=SimpleNamespace(build_reference_canvas_from_album=lambda _:b'album-reference',
            reference_cover_box=lambda *_:(461,205,1075,819),REFERENCE_BACKGROUND_PROMPT='Original prompt',
            ha_edit_to_frame=lambda _:Image.new('RGB',(24,16),'blue'))
        self.render_patch=mock.patch.object(pipeline,'render',return_value=self.root/'render.jpg');self.render_patch.start()
    def add_album(self,key):
        with store.db() as c:
            return c.execute("INSERT INTO albums(cache_key,artist,title,source_status,source_path,source_sha256,metadata) VALUES(?,?,?,'original','cover.png',?,'{}')",
                             (key,'Artist',key,self.sha)).lastrowid
    def tearDown(self):
        self.render_patch.stop()
        for patch in reversed(self.patches):patch.stop()
        self.temp.cleanup()
    def response(self,status=200):
        return SimpleNamespace(ok=status==200,status_code=status,headers={'x-request-id':'test'},text='rejected',json=lambda:self.payload)
    def run_plan(self,effects,overrides=None,workers=1):
        batch=queue_full_library({'stages':STAGES,'reference_overrides':overrides or {}},workers=workers)
        with mock.patch.object(pipeline,'post_image',side_effect=effects) as post:
            if workers==1:pipeline.run_worker(batch,'test-key',self.module)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures=[pool.submit(pipeline.run_worker,batch,'test-key',self.module) for _ in range(workers)]
                    for future in futures:future.result()
        with store.db() as c:rows=[dict(r) for r in c.execute('SELECT * FROM candidates ORDER BY id')]
        return rows,post
    def test_success_stops_and_retains_failed_attempts(self):
        rows,post=self.run_plan([self.response(400),self.response(400),self.response()])
        self.assertEqual([r['state'] for r in rows],['error','error','complete'])
        self.assertEqual([call.args[2]['model'] for call in post.call_args_list],[s['model'] for s in STAGES[:3]])
        self.assertTrue(all(any(f[0]=='mask' for f in call.args[1]) for call in post.call_args_list))
    def test_exhausted_ladder_is_bounded_and_legacy_has_no_mask(self):
        rows,post=self.run_plan([self.response(400)]*5)
        self.assertEqual(len(rows),5);self.assertEqual([r['stage'] for r in rows],list(range(5)))
        self.assertTrue(all(r['state']=='error' for r in rows))
        for call in post.call_args_list[3:]:
            self.assertEqual(call.args[2]['prompt'],'Original prompt')
            self.assertEqual([f[0] for f in call.args[1]],['image[]'])
    def test_uncertain_timeout_does_not_advance(self):
        rows,post=self.run_plan([requests.Timeout('Unknown outcome')])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['state'],'interrupted');self.assertEqual(post.call_count,1)
    def test_water_route_never_sends_original_cover_or_mask(self):
        water=b'water-only-reference';(self.root/'water.png').write_bytes(water)
        overrides={'a':{'path':'water.png','sha256':hashlib.sha256(water).hexdigest(),'source_sha256':self.sha}}
        rows,post=self.run_plan([self.response(400)]*3,overrides)
        self.assertEqual(len(rows),3)
        for call in post.call_args_list:
            self.assertEqual(call.args[1],[('image[]',('input.png',water,'image/png'))])
            self.assertEqual(call.args[2]['prompt'],pipeline.WATER_PROMPT)
    def test_parallel_claims_generate_each_album_only_once(self):
        self.add_album('b');self.add_album('c')
        rows,post=self.run_plan(lambda *_:self.response(),workers=3)
        self.assertEqual(post.call_count,3);self.assertEqual(len(rows),3)
        self.assertTrue(all(r['state']=='complete' for r in rows))
    def test_full_run_includes_but_does_not_approve_provisional_crops(self):
        with store.db() as c:c.execute("UPDATE albums SET source_status='extracted'")
        rows,_=self.run_plan([self.response()])
        self.assertEqual(rows[0]['state'],'complete')
        with store.db() as c:self.assertEqual(c.execute('SELECT source_status FROM albums').fetchone()[0],'extracted')
        with self.assertRaises(ValueError):queue_full_library({'stages':STAGES})
    def test_15_cost_uses_its_own_output_rate(self):
        self.assertAlmostEqual(pipeline.cost(self.payload['usage'],'gpt-image-1.5'),.00405)
    def test_visual_rejection_advances_on_explicit_regeneration(self):
        rows,_=self.run_plan([self.response()])
        store.review(rows[0]['id'],0,'rejected',['Not seamless'],'Visible boundary')
        batch=store.queue_batch([rows[0]['album_id']],'review retry')
        with store.db() as c:
            self.assertEqual(c.execute('SELECT mode FROM batches WHERE id=?',(batch,)).fetchone()[0],'fallback')
            self.assertEqual(c.execute('SELECT stage FROM candidates WHERE batch_id=?',(batch,)).fetchone()[0],1)

    def test_sunburst_request_uses_explicit_quality_and_frozen_prompt(self):
        stage={'model':'gpt-image-2.5-sunburst','mode':'masked','quality':'high'}
        batch=queue_full_library({'stages':[stage],'masked_prompt':'Continue the roof at the same slope'})
        with mock.patch.object(pipeline,'post_image',return_value=self.response()) as post:
            pipeline.run_worker(batch,'test-key',self.module)
        request=post.call_args.args[2]
        self.assertEqual(request['model'],'gpt-image-2.5-sunburst')
        self.assertEqual(request['quality'],'high')
        self.assertEqual(request['prompt'],'Continue the roof at the same slope')
        with store.db() as c:
            row=c.execute('SELECT * FROM candidates').fetchone()
            self.assertEqual(row['state'],'complete')
            self.assertEqual(json.loads(row['recipe'])['quality'],'high')

    def test_repair_sends_frozen_scene_with_opaque_center_mask_and_no_fallback(self):
        (self.root/'previews').mkdir();path=self.root/'previews/repair.png'
        Image.new('RGB',(1536,1024),'green').save(path);raw=path.read_bytes()
        stage={'model':'gpt-image-2.5-sunburst','mode':'repair','quality':'high'}
        ref=dict(path='previews/repair.png',sha256=hashlib.sha256(raw).hexdigest(),source_sha256=self.sha,prompt='Repair the fence')
        batch=queue_full_library({'stages':[stage],'repair_references':{'a':ref}})
        with mock.patch.object(pipeline,'post_image',return_value=self.response(400)) as post:
            pipeline.run_worker(batch,'test-key',self.module)
        files=dict(post.call_args.args[1]);self.assertEqual(files['image[]'][1],raw)
        mask=Image.open(BytesIO(files['mask'][1])).getchannel('A')
        self.assertEqual(mask.getpixel((461,205)),255);self.assertEqual(mask.getpixel((460,205)),0)
        self.assertEqual(post.call_args.args[2]['prompt'],'Repair the fence')
        with store.db() as c:self.assertEqual(c.execute('SELECT count(*) FROM candidates').fetchone()[0],1)

    def test_independent_background_never_uploads_cover_or_mask(self):
        batch=queue_full_library({'stages':[{'model':'gpt-image-2.5-sunburst','mode':'background','quality':'high'}],
                                 'background_prompts':{'a':'An empty blue sky'},'per_album_guidance':{'a':''}})
        with mock.patch.object(pipeline,'post_image',return_value=self.response()) as post:
            pipeline.run_worker(batch,'test-key',self.module)
        self.assertIsNone(post.call_args.args[1])
        self.assertEqual(post.call_args.args[2]['prompt'],'An empty blue sky')
        with store.db() as c:
            row=c.execute('SELECT * FROM candidates').fetchone()
            self.assertEqual(row['state'],'complete')
            self.assertEqual(json.loads(row['recipe'])['api_input'],'text_only')
            folder=self.root/row['folder']
            self.assertTrue((folder/'source/cover.png').exists())
            self.assertFalse((folder/'reference.png').exists())
            self.assertFalse((folder/'mask.png').exists())

    def test_background_transport_uses_json_generation_endpoint(self):
        with mock.patch.object(pipeline.requests,'post',return_value=self.response()) as post:
            pipeline.post_image('test-key',None,{'model':'gpt-image-2.5-sunburst','prompt':'Empty scenery'})
        self.assertTrue(post.call_args.args[0].endswith('/images/generations'))
        self.assertIn('json',post.call_args.kwargs)
        self.assertNotIn('files',post.call_args.kwargs)

    def test_changed_frozen_background_source_stops_before_api(self):
        batch=queue_full_library({'stages':[{'model':'gpt-image-2.5-sunburst','mode':'background','quality':'high'}],
                                 'background_prompts':{'a':'Empty scenery'},'source_checksums':{'a':'changed'}})
        with mock.patch.object(pipeline,'post_image') as post:
            pipeline.run_worker(batch,'test-key',self.module)
        post.assert_not_called()

if __name__=='__main__':unittest.main()
