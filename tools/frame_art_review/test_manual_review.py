import json
import unittest
from unittest import mock
import manual_review as manual
import store
import test_curation as fixtures


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        fixtures.CurationTests.setUp(self)
        p=mock.patch.object(manual,'ROOT',self.root);p.start();self.patches.append(p)
        manual.init()

    tearDown=fixtures.CurationTests.tearDown

    def test_flags_and_multiple_markers_survive_reload_without_changing_choices(self):
        manual.save(1,0,'original:1','flagged','1: roof, 2: shadow',
                    ['Edge / continuation mismatch'],[[.7,.1],[.2,.8]])
        manual.init()
        queue=manual.export()['queue']
        self.assertEqual(len(queue),1)
        self.assertEqual(queue[0]['points'],[[.7,.1],[.2,.8]])
        with store.db() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM album_selections').fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM second_pass_queue').fetchone()[0],0)
        with self.assertRaises(RuntimeError):manual.save(1,0,'original:1','ok')
        manual.save(1,1,'original:1','ok')
        self.assertEqual(manual.export()['queue'],[])

    def test_changed_art_reopens_review_and_rejects_stale_asset(self):
        manual.save(1,0,'original:1','ok')
        with store.db() as c:c.execute('UPDATE albums SET accepted_id=1 WHERE id=1')
        row=next(a for a in manual.state()['albums'] if a['id']==1)
        self.assertEqual(row['status'],'unreviewed')
        with self.assertRaises(RuntimeError):manual.save(1,1,'original:1','flagged')
        with self.assertRaises(ValueError):manual.save(1,1,'candidate:1','flagged',points=[[2,.2]])

    def test_release_art_used_until_selection_changes_and_duplicates_excluded(self):
        base=self.root/'release-original-finish';base.mkdir()
        (base/'approved.jpg').write_bytes(b'approved')
        (base/'release.json').write_text(json.dumps({'entries':[{'album_id':1,'asset':'original:1','files':{'compressed':{'path':'approved.jpg'}}}]}))
        self.assertEqual(manual.artwork(1).read_bytes(),b'approved')
        with store.db() as c:c.execute('UPDATE albums SET canonical_id=1 WHERE id=2')
        self.assertEqual(manual.state()['total'],2)
        with store.db() as c:c.execute('UPDATE albums SET accepted_id=1 WHERE id=1')
        with mock.patch.object(manual.curation,'render_asset',return_value=base/'new.jpg') as render:
            self.assertEqual(manual.artwork(1),base/'new.jpg')
            render.assert_called_once_with('candidate:1','original',0)


if __name__=='__main__':unittest.main()
