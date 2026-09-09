import json
import unittest
from io import BytesIO
from unittest import mock
from PIL import Image
import repairs
import store
import workflow
import test_curation as fixtures

class RepairTests(unittest.TestCase):
    setUp=fixtures.CurationTests.setUp
    tearDown=fixtures.CurationTests.tearDown

    def test_reference_retains_full_scene_and_exact_opaque_cover_placement(self):
        raw=repairs.reference_bytes(self.root/'snapshot/original.jpg',self.root/'snapshot/cover.png')
        im=Image.open(BytesIO(raw))
        self.assertEqual(im.size,(1536,1024))
        self.assertEqual(im.getpixel((461,205)),(255,0,0))
        self.assertEqual(im.getpixel((1074,818)),(255,0,0))
        self.assertEqual(im.getpixel((460,205)),im.getpixel((0,205)))

    def test_subset_queue_preserves_other_drafts_and_blocks_refusals_and_repeat(self):
        with store.db() as c:
            c.execute("UPDATE duplicate_groups SET status='separate'")
            c.execute("UPDATE batches SET plan='{}' WHERE id=4")
        for id in (1,2,3):workflow.decide(id,0,'regenerate')
        with store.db() as c:c.execute("UPDATE candidates SET state='error' WHERE album_id=2")
        spec=dict(album_id=2,revision=1,asset='original:2',instruction='Repair scene')
        with self.assertRaises(ValueError):repairs.queue_pilot([spec])
        spec.update(album_id=1,asset='original:1')
        batch=repairs.queue_pilot([spec])
        with store.db() as c:
            self.assertEqual([r[0] for r in c.execute('SELECT album_id FROM second_pass_queue ORDER BY album_id')],[2,3])
            self.assertEqual([r[0] for r in c.execute('SELECT album_id FROM candidates WHERE batch_id=?',(batch,))],[1])
            plan=json.loads(c.execute('SELECT plan FROM batches WHERE id=?',(batch,)).fetchone()[0])
            self.assertEqual(len(plan['stages']),1)
        with self.assertRaises(ValueError):repairs.queue_pilot([spec])

    def test_local_background_costs_zero_and_retains_all_prior_candidates(self):
        workflow.decide(3,0,'regenerate')
        with mock.patch.object(repairs.pipeline,'render'):
            id=repairs.solid_background(3,1,[1,1,1])
        with store.db() as c:
            row=c.execute('SELECT * FROM candidates WHERE id=?',(id,)).fetchone()
            self.assertEqual(row['cost'],0)
            self.assertEqual(row['state'],'complete')
            self.assertEqual(c.execute('SELECT count(*) FROM candidates WHERE album_id=3').fetchone()[0],2)
        self.assertEqual(workflow.detail(3)['full_review']['status'],'pending')

    def test_background_design_can_follow_refusal_without_resending_reference(self):
        workflow.decide(3,0,'regenerate')
        with store.db() as c:c.execute("UPDATE candidates SET state='error',error='moderation_blocked' WHERE album_id=3")
        spec=dict(album_id=3,revision=1,prompt='An empty blue sky with subtle natural cloud texture and no people or text.')
        batch=repairs.queue_backgrounds([spec],'Scenery test')
        with store.db() as c:
            plan=json.loads(c.execute('SELECT plan FROM batches WHERE id=?',(batch,)).fetchone()[0])
            self.assertEqual(plan['stages'][0]['mode'],'background')
            self.assertEqual(len(plan['stages']),1)
            self.assertNotIn('repair_references',plan)
            self.assertNotIn('reference_overrides',plan)
        with self.assertRaises(ValueError):repairs.queue_backgrounds([spec],'Scenery test')

    def test_background_queue_cannot_replace_confirmed_choice(self):
        workflow.decide(3,0,'choose',asset='original:3')
        with self.assertRaises(ValueError):repairs.queue_backgrounds([dict(album_id=3,revision=1,prompt='An empty blue sky with subtle natural cloud texture and no people or text.')],'Protected')
