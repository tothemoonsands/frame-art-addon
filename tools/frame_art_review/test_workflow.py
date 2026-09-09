import json
import unittest

import curation
import store
import workflow
import test_curation as fixtures


class WorkflowTests(unittest.TestCase):
    setUp=fixtures.CurationTests.setUp
    tearDown=fixtures.CurationTests.tearDown

    def test_catalog_fix_preserves_original_choice_and_snapshot_and_is_undoable(self):
        curation.choose(3,0,'original:3')
        workflow.decide(3,1,'fix')
        option=curation.source_crop(3)
        with store.db() as c:
            c.execute('UPDATE source_options SET origin=? WHERE id=?',(json.dumps(dict(method='verified_catalog_correction',collection_id=123,artist='Correct Artist',album='Correct Title')),option))
        result=workflow.correct_catalog(3,2,option)
        d=workflow.detail(3)
        self.assertEqual(d['album']['collection_id'],123)
        self.assertEqual(d['selection']['asset'],'original:3')
        with store.db() as c:self.assertEqual(c.execute('SELECT metadata FROM albums WHERE id=3').fetchone()[0],'{}')
        self.assertEqual(d['full_review']['status'],'pending')
        self.assertEqual(curation.reconciliation()['entries']['three']['collection_id'],123)
        workflow.undo(result['event_id'])
        self.assertIsNone(workflow.detail(3)['album']['collection_id'])
        self.assertEqual(workflow.detail(3)['full_review']['status'],'fix')

    def test_existing_kept_starts_pending_and_confirmation_is_undoable(self):
        store.review(3,0,'accepted',[],'Earlier decision')
        d=workflow.detail(3)
        self.assertEqual(d['full_review']['status'],'pending')
        result=workflow.decide(3,d['album']['revision'],'confirm',notes='Checked all versions')
        d=workflow.detail(3)
        self.assertEqual(d['full_review']['status'],'confirmed')
        self.assertEqual(d['selection']['asset'],'candidate:3')
        self.assertEqual(d['candidates'][0]['notes'],'Earlier decision')
        workflow.undo(result['event_id'])
        self.assertEqual(workflow.detail(3)['full_review']['status'],'pending')
        self.assertEqual(workflow.detail(3)['selection']['asset'],'candidate:3')

    def test_choose_original_auto_confirm_and_reject_stale_double_click(self):
        result=workflow.decide(3,0,'choose',asset='original:3')
        self.assertEqual(result['status'],'confirmed')
        self.assertEqual(workflow.detail(3)['selection']['asset'],'original:3')
        with self.assertRaises(RuntimeError):workflow.decide(3,0,'fix')
        with store.db() as c:self.assertEqual(c.execute('SELECT count(*) FROM workflow_events').fetchone()[0],1)

    def test_regeneration_stages_without_billing_and_undo_restores_kept_version(self):
        curation.choose(3,0,'original:3')
        result=workflow.decide(3,1,'regenerate',notes='Align the horizon',reasons=['Not seamless'])
        d=workflow.detail(3)
        self.assertIsNone(d['selection'])
        self.assertEqual(d['queued']['recipe'],'sunburst')
        self.assertIn('Align the horizon',d['queued']['guidance'])
        self.assertEqual(d['full_review']['status'],'regenerate')
        with store.db() as c:self.assertEqual(c.execute('SELECT count(*) FROM candidates').fetchone()[0],3)
        workflow.undo(result['event_id'])
        d=workflow.detail(3)
        self.assertEqual(d['selection']['asset'],'original:3')
        self.assertIsNone(d['queued'])

    def test_fix_flag_preserves_artwork_and_cannot_undo_over_later_choice(self):
        curation.choose(3,0,'original:3')
        flag=workflow.decide(3,1,'fix',notes='Wrong edition')
        self.assertEqual(workflow.detail(3)['selection']['asset'],'original:3')
        workflow.decide(3,2,'choose',asset='candidate:3')
        with self.assertRaises(RuntimeError):workflow.undo(flag['event_id'])

    def test_wrong_cover_queue_rolls_back_without_losing_choice(self):
        curation.choose(3,0,'original:3')
        with store.db() as c:c.execute('UPDATE candidates SET reasons=? WHERE id=3',('["Wrong album cover"]',))
        with self.assertRaises(ValueError):workflow.decide(3,1,'regenerate')
        self.assertEqual(workflow.detail(3)['selection']['asset'],'original:3')
        self.assertEqual(workflow.detail(3)['album']['revision'],1)

    def test_edit_cover_and_metadata_then_queue_is_atomic_and_undoable(self):
        option=curation.source_crop(3)
        result=workflow.edit(3,0,'Correct Artist','Correct Album',option,notes='Corrected cover',regenerate=True)
        d=workflow.detail(3)
        self.assertEqual(d['album']['artist'],'Correct Artist')
        self.assertEqual(d['album']['cache_key'],'three')
        self.assertEqual(d['album']['source_status'],'user_selected')
        self.assertEqual(d['queued']['source_sha256'],d['album']['source_sha256'])
        workflow.undo(result['event_id'])
        d=workflow.detail(3)
        self.assertEqual(d['album']['artist'],'Other')
        self.assertEqual(d['album']['source_sha256'],self.sha)
        self.assertIsNone(d['queued'])

    def test_manual_merge_and_undo_keep_every_record_and_version(self):
        result=workflow.merge(1,0,3,0,'candidate:2','Same album')
        d=workflow.detail(1)
        self.assertEqual(d['album']['id'],3)
        self.assertEqual(d['selection']['asset'],'candidate:2')
        self.assertEqual(len(d['assets']),6)
        self.assertEqual(d['full_review']['status'],'confirmed')
        self.assertEqual(curation.reconciliation()['alias_map']['one'],'three')
        workflow.undo(result['event_id'])
        self.assertEqual(workflow.detail(1)['album']['id'],1)
        self.assertEqual(workflow.detail(3)['album']['id'],3)
        with store.db() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM albums').fetchone()[0],3)
            self.assertEqual(c.execute('SELECT count(*) FROM candidates').fetchone()[0],3)

    def test_new_generation_invalidates_review_and_blocks_undo(self):
        result=workflow.decide(3,0,'choose',asset='original:3')
        with store.db() as c:c.execute("INSERT INTO candidates(album_id,batch_id,created_at,state) VALUES(3,4,?,'complete')",(store.now(),))
        self.assertEqual(workflow.detail(3)['full_review']['status'],'pending')
        with self.assertRaises(RuntimeError):workflow.undo(result['event_id'])

    def test_separate_tabs_cannot_merge_stale_target(self):
        workflow.decide(3,0,'later')
        with self.assertRaises(RuntimeError):workflow.merge(1,0,3,0,'original:1')
        self.assertEqual(workflow.detail(1)['album']['id'],1)

    def test_export_requires_full_review_and_does_not_confuse_fix_with_confirmed(self):
        with store.db() as c:c.execute("UPDATE duplicate_groups SET status='separate'")
        for id in (1,2,3):workflow.decide(id,0,'choose',asset=f'original:{id}')
        self.assertTrue(curation.reconciliation()['review_complete'])
        workflow.decide(3,1,'fix',notes='Check edition')
        plan=curation.reconciliation()
        self.assertFalse(plan['review_complete'])
        self.assertEqual(plan['full_review_remaining'],[3])
        self.assertEqual(plan['entries']['three']['selection']['asset'],'original:3')


if __name__=='__main__':unittest.main()
