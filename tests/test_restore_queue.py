import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Minimal stubs for optional runtime dependencies used by uploader.py imports.
if "PIL" not in sys.modules:
    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = object()
    sys.modules["PIL"] = pil_mod
if "samsungtvws" not in sys.modules:
    stv = types.ModuleType("samsungtvws")
    stv.SamsungTVWS = object
    sys.modules["samsungtvws"] = stv
if "cover_art" not in sys.modules:
    cover = types.ModuleType("cover_art")
    cover.SOURCE_DIR = Path(".")
    cover.WIDESCREEN_DIR = Path(".")
    cover.download_artwork = lambda *a, **k: None
    cover.ensure_dirs = lambda *a, **k: None
    cover.itunes_lookup = lambda *a, **k: {}
    cover.itunes_search = lambda *a, **k: {}
    cover.normalize_key = lambda *a, **k: "k"
    cover.generate_reference_frame_from_album = lambda *a, **k: (b"", b"", None, None)
    cover.resolve_artwork_url = lambda *a, **k: ""
    sys.modules["cover_art"] = cover

import frame_art_uploader_ai.uploader as uploader


class RestoreQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inbox = self.root / "frame_art_restore_request.json"
        self.queue = self.root / "frame_art_restore_queue"
        self.lock = self.root / "frame_art_uploader_worker.lock"

        uploader.RESTORE_REQUEST_PATH = str(self.inbox)
        uploader.RESTORE_QUEUE_DIR = self.queue
        uploader.WORKER_LOCK_PATH = self.lock

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_inbox(self, payload):
        self.inbox.write_text(json.dumps(payload), encoding="utf-8")

    def test_two_immediate_successive_requests_fifo(self):
        self._write_inbox({"kind": "restore", "value": "A", "requested_at": "2026-01-01T00:00:00+00:00"})
        first = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(first)

        self._write_inbox({"kind": "cover_art_reference", "value": "B", "requested_at": "2026-01-01T00:00:01+00:00"})
        second = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(second)

        queued = uploader.list_queued_requests()
        self.assertEqual(2, len(queued))
        first_payload = json.loads(queued[0].read_text(encoding="utf-8"))
        second_payload = json.loads(queued[1].read_text(encoding="utf-8"))
        self.assertEqual("A", first_payload["value"])
        self.assertEqual("B", second_payload["value"])

    def test_three_rapid_requests(self):
        for idx in range(3):
            self._write_inbox({"kind": "pick", "value": f"R{idx}", "requested_at": f"2026-01-01T00:00:0{idx}+00:00"})
            uploader.enqueue_restore_inbox_if_present()

        queued = uploader.list_queued_requests()
        self.assertEqual(3, len(queued))
        values = [json.loads(p.read_text(encoding="utf-8"))["value"] for p in queued]
        self.assertEqual(["R0", "R1", "R2"], values)

    def test_malformed_followed_by_valid(self):
        self.inbox.write_text("{not-json", encoding="utf-8")
        uploader.enqueue_restore_inbox_if_present()

        self._write_inbox({"kind": "content_id", "value": "MY_F1"})
        uploader.enqueue_restore_inbox_if_present()

        queued = uploader.list_queued_requests()
        self.assertEqual(2, len(queued))

        p1, show1, err1 = uploader.load_restore_work_item(queued[0])
        self.assertIsNone(p1)
        self.assertIsNotNone(err1)
        self.assertIsNone(show1)

        p2, show2, err2 = uploader.load_restore_work_item(queued[1])
        self.assertIsNone(err2)
        self.assertEqual("content_id", p2["kind"])
        self.assertIsNone(show2)

    def test_worker_lock_single_holder(self):
        with uploader.worker_lock() as lock1:
            self.assertTrue(lock1)
            with uploader.worker_lock() as lock2:
                self.assertFalse(lock2)


if __name__ == "__main__":
    unittest.main()
