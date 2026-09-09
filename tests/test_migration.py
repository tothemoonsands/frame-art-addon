import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image
from frame_art_uploader_ai import migration as m


class TV:
    def __init__(self):
        self.ids = {'MY_F1','MY_FKEEP','SAM_OTHER'}
        self.current = 'MY_F1'
        self.calls = []
        self.timeout_upload = False

    def available(self):
        return [{'content_id': cid} for cid in sorted(self.ids)]

    def get_current(self):
        return {'content_id': self.current}

    def select_image(self, cid, **kwargs):
        self.calls.append(('select',cid))
        self.current = cid

    def delete(self, cid):
        self.calls.append(('delete',cid))
        self.ids.remove(cid)

    def upload(self, data, **kwargs):
        self.calls.append(('upload',data))
        self.ids.add('MY_FNEW')
        if self.timeout_upload:
            raise TimeoutError('lost acknowledgement')
        return 'MY_FNEW'

    def get_thumbnail(self,cid):
        out = BytesIO()
        Image.new('RGB',(40,20)).save(out,format='JPEG')
        return out.getvalue()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.release = self.root/'release'
        self.music = self.root/'music'
        self.share = self.root/'share'
        self.data = self.root/'data'
        for path in (self.release,self.music/'widescreen-compressed',self.share,self.data):
            path.mkdir(parents=True)
        Image.new('RGB',(3840,2160),'blue').save(self.release/'new.jpg')
        (self.music/'widescreen-compressed/old.jpg').write_bytes(b'old backup bytes')
        (self.music/'widescreen-compressed/keep.jpg').write_bytes(b'retained bytes')
        catalog={'entries':{'old.jpg':{'content_id':'MY_F1'},'keep.jpg':{'content_id':'MY_FKEEP'}}}
        m.write(self.share/m.META[0],catalog)
        m.write(self.share/m.META[4],{'entries':{}})
        m.write(self.share/m.META[5],{'entries':{}})
        m.write(self.data/'frame_art_uploader_state.json',{'last_music_content_id':'MY_F1'})
        self.row=dict(album_id=1,key='correct',old_key='wrong',lookup_keys=['correct'],text_keys=['Artist — Album'],
            artist='Artist',title='Album',collection_id=20,asset='candidate:1',recipe={},finish={},catalog_key='new.jpg',
            reuse_id=None,old_ids=['MY_F1'],old_files={'MY_F1':'old.jpg'},files={
                kind:dict(path='new.jpg',sha256=m.sha(self.release/'new.jpg'),target=f'{kind}/new.jpg')
                for kind in ['source','recipe','compressed']})
        self.row['files']['compressed']['target']='widescreen-compressed/new.jpg'
        self.keep=dict(self.row,album_id=2,key='keep',lookup_keys=['keep'],text_keys=['Artist — Keep'],
            title='Keep',catalog_key='keep.jpg',reuse_id='MY_FKEEP',old_ids=[],old_files={},files={
                k:dict(v,target='widescreen-compressed/keep.jpg' if k=='compressed' else 'keep/'+k+'.jpg') for k,v in self.row['files'].items()})
        self.doc=dict(entries=[self.row,self.keep],expected_catalog=m.active_map(catalog),wrong_collection_keys=['wrong'])
        m.write(self.release/'release.json',self.doc)
        self.control=self.root/'active.json'
        m.write(self.control,dict(release=str(self.release),run=str(self.root/'run'),paused=False))
        self.tv=TV()

    def tearDown(self):
        self.temp.cleanup()

    def worker(self):
        return m.Migration(self.control,self.tv,music=self.music,share=self.share,data=self.data,sleep=lambda _:None)

    def test_delete_first_verified_mapping_and_unrelated_art_preserved(self):
        worker=self.worker()
        worker.execute()
        actions=[x[0] for x in self.tv.calls]
        self.assertLess(actions.index('delete'),actions.index('upload'))
        self.assertEqual(1,actions.count('upload'))
        self.assertEqual((self.release/'new.jpg').read_bytes(),next(x[1] for x in self.tv.calls if x[0]=='upload'))
        self.assertEqual({'MY_FNEW','MY_FKEEP','SAM_OTHER'},self.tv.ids)
        self.assertEqual('MY_FNEW',m.read(self.share/m.META[0])['entries']['new.jpg']['content_id'])
        self.assertEqual('MY_FNEW',m.read(self.data/'frame_art_uploader_state.json')['last_music_content_id'])
        self.assertFalse(self.control.exists())
        self.assertTrue((self.root/'run/backup/music/widescreen-compressed/old.jpg').exists())

    def test_uncertain_upload_never_replayed_or_followed_by_deletion(self):
        self.tv.timeout_upload=True
        with self.assertRaises(TimeoutError):
            self.worker().execute()
        self.tv.timeout_upload=False
        with self.assertRaisesRegex(m.Paused,'acknowledgement uncertain'):
            self.worker().execute()
        self.assertEqual(1,sum(x[0]=='upload' for x in self.tv.calls))
        self.assertEqual(1,sum(x[0]=='delete' for x in self.tv.calls))
        self.assertTrue(self.control.exists())

    def test_resume_after_verified_upload_commits_without_reupload(self):
        worker=self.worker()
        worker.preflight()
        original=worker.commit_row
        worker.commit_row=lambda *args: (_ for _ in ()).throw(OSError('disk unavailable'))
        with self.assertRaises(OSError):worker.replace(self.row)
        worker=self.worker()
        worker.execute()
        self.assertEqual(1,sum(x[0]=='upload' for x in self.tv.calls))

    def test_changed_release_file_stops_before_tv_mutations(self):
        (self.release/'new.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.worker().execute()
        self.assertEqual([],self.tv.calls)

    def test_live_catalog_drift_stops_before_tv_mutations(self):
        doc=m.read(self.share/m.META[0]);doc['entries']['new-album.jpg']={'content_id':'MY_F9'}
        m.write(self.share/m.META[0],doc)
        with self.assertRaisesRegex(ValueError,'catalog changed'):
            self.worker().execute()
        self.assertEqual([],self.tv.calls)

    def test_wrong_collection_alias_removed_and_duplicate_preserved(self):
        old={'entries':{'cache::wrong':{'content_id':'MY_F1'},'session::samealbum':{'content_id':'MY_F1'}}}
        backup=self.root/'backup'
        m.write(backup/'share/frame_art_music_associations.json',old)
        docs=m.reconcile(self.doc,{'1':{'content_id':'MY_FNEW'},'2':{'content_id':'MY_FKEEP'}},backup,self.music,self.share)
        entries=docs[str(self.share/m.META[1])]['entries']
        self.assertNotIn('cache::wrong',entries)
        self.assertEqual('MY_FNEW',entries['cache::correct']['content_id'])
        self.assertEqual('MY_FNEW',entries['session::samealbum']['content_id'])


if __name__=='__main__':unittest.main()
