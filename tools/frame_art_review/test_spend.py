import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import store
import pipeline

class SpendTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.patch=mock.patch.object(store,'ROOT',Path(self.temp.name).resolve());self.patch.start()
        store.init()
        with store.db() as c:
            self.album=c.execute("INSERT INTO albums(cache_key,artist,title,metadata) VALUES('a','Artist','Album','{}')").lastrowid
            self.batch=c.execute("INSERT INTO batches(label,created_at,paused,spend_limit) VALUES('test',?,1,50)",(store.now(),)).lastrowid
        self.old=self.candidate('complete',50,0)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def candidate(self,state,cost,run):
        with store.db() as c:
            return c.execute('INSERT INTO candidates(album_id,batch_id,created_at,state,cost,spend_run) VALUES(?,?,?,?,?,?)',
                             (self.album,self.batch,store.now(),state,cost,run)).lastrowid
    def test_resume_resets_only_run_and_keeps_lifetime_history(self):
        run=store.resume_batch(self.batch)
        self.assertEqual(run,1)
        with store.db() as c:
            self.assertEqual(store.guard_spend(c,self.batch,run),0)
            self.assertEqual(store.guard_spend(c),50)
            self.assertEqual(c.execute('SELECT cost FROM candidates WHERE id=?',(self.old,)).fetchone()[0],50)
        self.candidate('complete',4,run)
        self.assertEqual(store.resume_batch(self.batch),run)
        with store.db() as c:
            self.assertEqual(store.guard_spend(c,self.batch,run),4)
            c.execute('UPDATE batches SET paused=1 WHERE id=?',(self.batch,))
        self.assertEqual(store.resume_batch(self.batch),2)
        with store.db() as c:
            self.assertEqual(store.guard_spend(c,self.batch,2),0)
            self.assertEqual(store.guard_spend(c),54)
            self.assertEqual(c.execute('SELECT count(*) FROM spend_runs').fetchone()[0],2)
    def test_guard_pauses_before_next_request(self):
        run=store.resume_batch(self.batch)
        self.candidate('complete',49.6,run);queued=self.candidate('queued',None,0)
        with mock.patch.object(pipeline,'post_image') as post:
            pipeline.run_worker(self.batch,'test-key',SimpleNamespace());post.assert_not_called()
        with store.db() as c:
            self.assertEqual(c.execute('SELECT state FROM candidates WHERE id=?',(queued,)).fetchone()[0],'queued')
            self.assertEqual(c.execute('SELECT paused FROM batches WHERE id=?',(self.batch,)).fetchone()[0],1)
    def test_lifetime_cap_cannot_be_reset_by_resume(self):
        self.candidate('complete',150,0)
        with self.assertRaisesRegex(ValueError,'Lifetime'):store.resume_batch(self.batch)
        with store.db() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM spend_runs').fetchone()[0],0)
            self.assertEqual(store.guard_spend(c),200)
    def test_lifetime_cap_blocks_worker_with_fresh_run(self):
        run=store.resume_batch(self.batch)
        self.candidate('complete',150,0);self.candidate('queued',None,run)
        with mock.patch.object(pipeline,'post_image') as post:
            pipeline.run_worker(self.batch,'test-key',SimpleNamespace());post.assert_not_called()
        with store.db() as c:self.assertIn('Lifetime',c.execute('SELECT pause_reason FROM batches WHERE id=?',(self.batch,)).fetchone()[0])
    def test_previous_inflight_settlement_does_not_consume_new_run(self):
        old_active=self.candidate('running',None,0);run=store.resume_batch(self.batch)
        with store.db() as c:
            c.execute("UPDATE candidates SET state='complete',cost=.2 WHERE id=?",(old_active,))
            self.assertEqual(store.guard_spend(c,self.batch,run),0)
            self.assertAlmostEqual(store.guard_spend(c),50.2)

if __name__=='__main__':unittest.main()
