import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import store

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.patch=mock.patch.object(store,'ROOT',Path(self.temp.name));self.patch.start()
        store.init()
        with store.db() as c:
            self.album=c.execute("INSERT INTO albums(cache_key,artist,title,source_status,metadata) VALUES('test','Artist','Album','original','{}')").lastrowid
        self.batch=store.queue_batch([self.album],'test')
        with store.db() as c:
            self.candidate=c.execute('SELECT id FROM candidates').fetchone()[0]
            c.execute("UPDATE candidates SET state='complete' WHERE id=?",(self.candidate,))
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_accept_persists_and_blocks_regeneration(self):
        store.review(self.candidate,0,'accepted',[],'Keep this')
        with store.db() as c:
            self.assertEqual(c.execute('SELECT accepted_id FROM albums').fetchone()[0],self.candidate)
            self.assertEqual(c.execute('SELECT notes FROM candidates').fetchone()[0],'Keep this')
        with self.assertRaises(ValueError):store.queue_batch([self.album],'not allowed')
    def test_stale_tab_cannot_overwrite_review(self):
        store.review(self.candidate,0,'accepted',[],'Keep')
        with self.assertRaises(RuntimeError):store.review(self.candidate,0,'rejected',['Not seamless'],'stale')
        with store.db() as c:self.assertEqual(c.execute('SELECT decision FROM candidates').fetchone()[0],'accepted')
    def test_rejection_requires_reason_without_partial_write(self):
        with self.assertRaises(ValueError):store.review(self.candidate,0,'rejected',[],'')
        with store.db() as c:self.assertEqual(c.execute('SELECT count(*) FROM review_events').fetchone()[0],0)
    def test_rejected_retry_keeps_prior_candidate_and_reason(self):
        store.review(self.candidate,0,'rejected',['Not seamless'],'visible boundary')
        batch2=store.queue_batch([self.album],'retry')
        with store.db() as c:
            rows=c.execute('SELECT * FROM candidates ORDER BY id').fetchall()
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['notes'],'visible boundary')
            self.assertEqual(rows[1]['batch_id'],batch2);self.assertEqual(rows[1]['state'],'queued')
        with self.assertRaises(ValueError):store.queue_batch([self.album],'duplicate')
    def test_wrong_cover_cannot_blindly_retry(self):
        store.review(self.candidate,0,'rejected',['Wrong album cover'],'')
        with self.assertRaises(ValueError):store.queue_batch([self.album],'wrong source')
    def test_clear_accept_releases_pin_and_preserves_events(self):
        store.review(self.candidate,0,'accepted',[],'')
        store.review(self.candidate,1,'unreviewed',[],'Reconsider')
        with store.db() as c:
            self.assertIsNone(c.execute('SELECT accepted_id FROM albums').fetchone()[0])
            self.assertEqual(c.execute('SELECT count(*) FROM review_events').fetchone()[0],2)
    def test_accepting_different_version_keeps_only_one_pin(self):
        store.review(self.candidate,0,'rejected',['Not seamless'],'')
        store.queue_batch([self.album],'retry')
        with store.db() as c:
            newer=c.execute('SELECT max(id) FROM candidates').fetchone()[0]
            c.execute("UPDATE candidates SET state='complete' WHERE id=?",(newer,))
        store.review(self.candidate,1,'accepted',[],'Changed mind')
        store.review(newer,0,'accepted',[],'Prefer newer')
        with store.db() as c:
            self.assertEqual(c.execute("SELECT count(*) FROM candidates WHERE decision='accepted'").fetchone()[0],1)
            self.assertEqual(c.execute('SELECT accepted_id FROM albums').fetchone()[0],newer)
    def test_interrupted_request_cannot_be_recharged_implicitly(self):
        with store.db() as c:c.execute("UPDATE candidates SET state='interrupted'")
        with self.assertRaises(ValueError):store.queue_batch([self.album],'unsafe retry')
    def test_local_file_routes_do_not_expose_database_or_other_files(self):
        for path in ('../outside.png','review.sqlite3','snapshot/../../outside.jpg','snapshot/info.json'):
            with self.assertRaises(ValueError):store.safe_file(path)
    def test_sqlite_backup_keeps_review(self):
        store.review(self.candidate,0,'accepted',[],'Back me up')
        path=store.backup()
        import sqlite3
        with sqlite3.connect(path) as c:self.assertEqual(c.execute('SELECT notes FROM candidates').fetchone()[0],'Back me up')

if __name__=='__main__':unittest.main()
