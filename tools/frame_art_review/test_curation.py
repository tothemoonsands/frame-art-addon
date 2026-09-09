import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops
import curation
import pipeline
import store


class CurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.patches = [mock.patch.object(module, 'ROOT', self.root)
                        for module in (store, pipeline, curation)]
        for patch in self.patches: patch.start()
        store.init()
        (self.root / 'snapshot').mkdir()
        Image.new('RGB', (3840, 2160), 'navy').save(self.root / 'snapshot/original.jpg')
        Image.new('RGB', (300, 300), 'red').save(self.root / 'snapshot/cover.png')
        self.sha = hashlib.sha256((self.root / 'snapshot/cover.png').read_bytes()).hexdigest()
        with store.db() as c:
            for key, artist, title in [('one', 'Ye', 'BULLY - EP'), ('two', 'Kanye West', 'Bully'), ('three', 'Other', 'Different')]:
                c.execute('INSERT INTO albums(cache_key,artist,title,current_path,source_path,source_sha256,source_status,metadata) VALUES(?,?,?,?,?,?,?,?)',
                          (key, artist, title, 'snapshot/original.jpg', 'snapshot/cover.png', self.sha, 'original', '{}'))
            c.execute("INSERT INTO batches(id,label,created_at,mode,plan) VALUES(4,'First pass',?,'fallback',?)",
                      (store.now(), json.dumps({'stages': curation.STAGES, 'reference_overrides': {'three': {'path': 'snapshot/water.png', 'source_sha256': self.sha}}})))
            for album in (1, 2, 3):
                c.execute("INSERT INTO candidates(album_id,batch_id,state,created_at,recipe,decision,reasons,notes) VALUES(?,4,'complete',?,?,'rejected',?,?)",
                          (album, store.now(), json.dumps({'source_sha256': self.sha, 'feather_px': 24}), '["Not seamless"]', 'Original is better'))
        curation.discover_duplicates()
        curation.prepare_sources()

    def tearDown(self):
        for patch in reversed(self.patches): patch.stop()
        self.temp.cleanup()

    def group_args(self):
        d = curation.detail(1)
        return dict(group_id=d['groups'][0]['id'], revision=d['groups'][0]['revision'],
                    member_revisions={str(m['id']): m['revision'] for m in d['members']})

    def test_original_choice_preserves_reviews_and_blocks_generation(self):
        store.review(3, 0, 'accepted', [], 'Keep until compared')
        d = curation.detail(3)
        curation.choose(3, d['album']['revision'], 'original:3', 'strong', 24)
        d = curation.detail(3)
        self.assertEqual(d['selection']['asset'], 'original:3')
        self.assertEqual(d['selection']['shadow'], 'none')
        self.assertEqual(d['candidates'][0]['notes'], 'Keep until compared')
        self.assertIsNone(d['album']['accepted_id'])
        with self.assertRaises(ValueError): store.queue_batch([3], 'must not generate')
        with self.assertRaises(ValueError): store.review(3, 1, 'rejected', ['Not seamless'], 'stale page')
        with self.assertRaises(RuntimeError): curation.choose(3, 0, 'candidate:3')

    def test_duplicate_merge_preserves_versions_and_exports_aliases(self):
        curation.resolve_group(**self.group_args(), action='merge', canonical_id=1, asset='candidate:2', shadow='strong', feather=12)
        d = curation.detail(2)
        self.assertEqual(d['album']['id'], 1)
        self.assertEqual(d['selection']['asset'], 'candidate:2')
        self.assertEqual(len(d['assets']), 4)
        self.assertTrue(all(c['decision'] == 'rejected' for c in d['candidates']))
        plan = curation.reconciliation()
        self.assertEqual(plan['alias_map']['two'], 'one')
        self.assertNotIn('two', plan['entries'])
        self.assertIn('two', plan['original_metadata_by_key'])
        self.assertFalse(plan['deployment_ready'])
        curation.resolve_group(**self.group_args(), action='undo')
        self.assertEqual(curation.detail(2)['album']['id'], 2)
        self.assertTrue((self.root / 'snapshot/original.jpg').exists())

    def test_concurrent_album_choice_invalidates_duplicate_merge(self):
        stale = self.group_args()
        curation.choose(2, 0, 'original:2')
        with self.assertRaises(RuntimeError):
            curation.resolve_group(**stale, action='merge', canonical_id=1, asset='original:1')
        self.assertEqual(curation.detail(2)['selection']['asset'], 'original:2')

    def test_wrong_cover_requires_a_different_source(self):
        with store.db() as c: c.execute('UPDATE candidates SET reasons=? WHERE album_id=3', ('["Wrong album cover"]',))
        with self.assertRaises(ValueError): curation.stage_second_pass(3, 0, 'fallback')
        cropped = curation.source_crop(3)
        curation.select_source(3, 0, cropped)
        curation.stage_second_pass(3, 1, 'original-flare', 'Match the corrected cover')
        self.assertEqual(curation.detail(3)['queued']['guidance'], 'Match the corrected cover')
        with self.assertRaises(ValueError): curation.select_source(1, 0, cropped)

    def test_draft_is_unbilled_and_launch_preserves_recipe_guidance_and_water_route(self):
        curation.stage_second_pass(3, 0, 'original-flare', 'Extend the water naturally')
        with store.db() as c: self.assertEqual(c.execute('SELECT count(*) FROM candidates').fetchone()[0], 3)
        batch_id = curation.launch_second_pass()
        with store.db() as c:
            batch = c.execute('SELECT * FROM batches WHERE id=?', (batch_id,)).fetchone()
            self.assertEqual(c.execute('SELECT count(*) FROM second_pass_queue').fetchone()[0], 0)
            self.assertEqual(c.execute("SELECT count(*) FROM candidates WHERE state='queued'").fetchone()[0], 1)
        plan = json.loads(batch['plan'])
        self.assertEqual(plan['per_album_stages']['three'], [{'model': 'gpt-image-2.5-flare', 'mode': 'water'}])
        self.assertEqual(plan['per_album_guidance']['three'], 'Extend the water naturally')
        self.assertEqual(batch['spend_limit'], 50)

    def test_unresolved_duplicates_and_changed_sources_block_launch(self):
        with self.assertRaises(ValueError): curation.stage_second_pass(1, 0, 'fallback')
        curation.stage_second_pass(3, 0, 'fallback')
        (self.root / 'snapshot/cover.png').write_bytes(b'changed')
        with self.assertRaises(ValueError): curation.launch_second_pass()
        self.assertIsNotNone(curation.detail(3)['queued'])

    def test_explicit_full_draft_launch_keeps_spend_guards(self):
        with store.db() as c:
            for number in range(27):
                c.execute('INSERT INTO albums(cache_key,artist,title,source_path,source_sha256,source_status,metadata) VALUES(?,?,?,?,?,?,?)',
                          (f'bulk-{number}', 'Bulk', str(number), 'snapshot/cover.png', self.sha, 'original', '{}'))
        for album_id in range(4, 31): curation.stage_second_pass(album_id, 0, 'fallback', 'Match the scene')
        batch_id = curation.launch_second_pass(limit=None)
        with store.db() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM candidates WHERE batch_id=?', (batch_id,)).fetchone()[0], 27)
            self.assertEqual(c.execute('SELECT count(*) FROM second_pass_queue').fetchone()[0], 0)
            self.assertEqual(c.execute('SELECT spend_limit FROM batches WHERE id=?', (batch_id,)).fetchone()[0], 50)
            self.assertEqual(c.execute('SELECT lifetime_limit FROM spend_settings').fetchone()[0], 200)

    def test_keep_removes_draft_and_reopened_group_blocks_launch(self):
        curation.stage_second_pass(3, 0, 'fallback')
        curation.choose(3, 0, 'original:3')
        self.assertIsNone(curation.detail(3)['queued'])
        curation.resolve_group(**self.group_args(), action='separate')
        curation.stage_second_pass(1, 1, 'fallback')
        with store.db() as c: c.execute("UPDATE duplicate_groups SET status='open'")
        with self.assertRaises(ValueError): curation.launch_second_pass()

    def test_original_is_byte_identical_and_feather_changes_only_cover_edges(self):
        original = self.root / 'snapshot/original.jpg'
        self.assertEqual(curation.render_asset('original:1', 'strong', 24).read_bytes(), original.read_bytes())
        bg = Image.new('RGB', (3840, 2160), 'navy')
        solid = pipeline.composite(bg, self.root / 'snapshot/cover.png', 'regular', 0)
        feathered = pipeline.composite(bg, self.root / 'snapshot/cover.png', 'regular', 24)
        bbox = ImageChops.difference(solid, feathered).getbbox()
        self.assertEqual(bbox, (1152, 312, 2688, 1848))
        self.assertIsNone(ImageChops.difference(solid.crop((1176, 336, 2664, 1824)), feathered.crop((1176, 336, 2664, 1824))).getbbox())
        self.assertIsNone(ImageChops.difference(solid.crop((0, 0, 1152, 2160)), feathered.crop((0, 0, 1152, 2160))).getbbox())

    def test_global_finish_applies_to_existing_and_future_choices_and_export(self):
        curation.choose(1, 0, 'candidate:1')
        curation.choose(2, 0, 'original:2')
        store.review(3, 0, 'accepted', [], 'Keep')
        curation.save_render_style('strong', 12, 0)
        for album_id in (1, 3):
            choice = curation.detail(album_id)['selection']
            self.assertEqual((choice['shadow'], choice['feather']), ('strong', 12))
        original = curation.detail(2)['selection']
        self.assertEqual((original['shadow'], original['feather']), ('none', 0))
        plan = curation.reconciliation()
        self.assertEqual(plan['render_style']['shadow'], 'strong')
        self.assertEqual(plan['entries']['one']['selection']['feather'], 12)
        # Old tabs cannot reintroduce an individual rendering style when choosing a version.
        curation.choose(1, 1, 'candidate:1', 'none', 0)
        self.assertEqual(curation.detail(1)['selection']['shadow'], 'strong')
        with self.assertRaises(RuntimeError): curation.save_render_style('none', 0, 0)
        with store.db() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM candidates').fetchone()[0], 3)
            self.assertEqual(c.execute('SELECT notes FROM candidates WHERE id=3').fetchone()[0], 'Keep')

    def test_sunburst_restart_preserves_history_and_skips_kept_artwork(self):
        curation.resolve_group(**self.group_args(), action='separate')
        curation.choose(2, 1, 'original:2')
        with store.db() as c:
            c.execute('UPDATE batches SET paused=1 WHERE id=4')
            c.execute("UPDATE candidates SET state='queued' WHERE album_id=1")
        batch_id=curation.restart_with_sunburst(4,{1:'Continue the roof diagonal'})
        with store.db() as c:
            prior=c.execute('SELECT * FROM candidates WHERE id=1').fetchone()
            self.assertEqual(prior['state'],'canceled')
            self.assertEqual(prior['notes'],'Original is better')
            self.assertEqual(c.execute('SELECT state FROM candidates WHERE id=3').fetchone()[0],'complete')
            self.assertEqual([r[0] for r in c.execute('SELECT album_id FROM candidates WHERE batch_id=? ORDER BY album_id',(batch_id,))],[1,3])
            b=c.execute('SELECT * FROM batches WHERE id=?',(batch_id,)).fetchone()
        p=json.loads(b['plan'])
        self.assertEqual(p['per_album_stages']['one'],[{'model':'gpt-image-2.5-sunburst','mode':'masked','quality':'high'}])
        self.assertEqual(p['per_album_stages']['three'][0]['mode'],'water')
        self.assertIn('Continue the roof diagonal',p['per_album_guidance']['one'])
        self.assertIn('EXISTING GRAPHIC BORDERS',p['masked_prompt'])
        self.assertEqual(b['paused'],1)
        with self.assertRaises(ValueError):curation.restart_with_sunburst(4)


if __name__ == '__main__': unittest.main()
