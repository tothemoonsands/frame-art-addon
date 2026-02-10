import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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


class AmbientSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ambient = self.root / "ambient"
        self.catalog = self.root / "ambient_catalog.json"
        self.ambient.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parse_dispatch_ambient_seed(self):
        payload = {
            "kind": "ambient_seed",
            "show": False,
            "ambient_dir": str(self.ambient),
            "catalog_path": str(self.catalog),
            "force_reupload": True,
        }
        normalized, requested_show, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertEqual("ambient_seed", normalized["kind"])
        self.assertFalse(requested_show)
        self.assertEqual(str(self.ambient), normalized["ambient_dir"])
        self.assertEqual(str(self.catalog), normalized["catalog_path"])
        self.assertTrue(normalized["force_reupload"])

    def test_empty_dir_errors_clearly(self):
        with self.assertRaisesRegex(ValueError, "no supported images"):
            uploader.handle_ambient_seed_restore(
                tv_ip="127.0.0.1",
                art=object(),
                restore_payload={
                    "kind": "ambient_seed",
                    "ambient_dir": str(self.ambient),
                    "catalog_path": str(self.catalog),
                    "force_reupload": False,
                },
                requested_at="",
            )

    @mock.patch("frame_art_uploader_ai.uploader.upload_local_file_with_reconnect")
    def test_happy_path_upload_updates_catalog(self, mocked_upload):
        (self.ambient / "b.webp").write_bytes(b"b")
        (self.ambient / "a.JPG").write_bytes(b"a")

        def _uploader(tv_ip, art, file_path):
            return art, f"CID-{file_path.name}"

        mocked_upload.side_effect = _uploader

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=object(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": False,
            },
            requested_at="",
        )

        self.assertTrue(status["ok"])
        self.assertEqual(2, status["uploaded_count"])
        self.assertEqual(0, status["skipped_count"])
        catalog = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("CID-a.JPG", catalog["entries"]["a.JPG"]["content_id"])
        self.assertEqual("CID-b.webp", catalog["entries"]["b.webp"]["content_id"])

    @mock.patch("frame_art_uploader_ai.uploader.upload_local_file_with_reconnect")
    def test_nested_directories_are_scanned_and_cataloged_by_relative_path(self, mocked_upload):
        nested = self.ambient / "fall" / "night"
        nested.mkdir(parents=True)
        image = nested / "image.jpg"
        image.write_bytes(b"x")
        mocked_upload.return_value = (object(), "CID-NESTED")

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=object(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": False,
            },
            requested_at="",
        )

        self.assertTrue(status["ok"])
        self.assertEqual(1, status["uploaded_count"])
        catalog = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("CID-NESTED", catalog["entries"]["fall/night/image.jpg"]["content_id"])

    @mock.patch("frame_art_uploader_ai.uploader.upload_local_file_with_reconnect")
    def test_nested_files_reuse_legacy_basename_catalog_entry(self, mocked_upload):
        nested = self.ambient / "fall" / "night"
        nested.mkdir(parents=True)
        image = nested / "image.jpg"
        image.write_bytes(b"x")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "image.jpg": {"content_id": "LEGACY", "updated_at": ""},
                },
            },
        )

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=object(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": False,
            },
            requested_at="",
        )

        self.assertEqual(0, status["uploaded_count"])
        self.assertEqual(1, status["skipped_count"])
        self.assertEqual(0, mocked_upload.call_count)

    @mock.patch("frame_art_uploader_ai.uploader.upload_local_file_with_reconnect")
    def test_force_reupload_bypasses_cached_ids(self, mocked_upload):
        (self.ambient / "a.png").write_bytes(b"a")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "a.png": {"content_id": "CACHED", "updated_at": ""},
                },
            },
        )
        mocked_upload.return_value = (object(), "NEWCID")

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=object(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": True,
            },
            requested_at="",
        )

        self.assertEqual(1, status["uploaded_count"])
        self.assertEqual(0, status["skipped_count"])
        self.assertEqual(1, mocked_upload.call_count)

    def test_non_ambient_kinds_unaffected(self):
        payload = {"kind": "pick", "phase": "night", "season": "winter"}
        normalized, requested_show, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertEqual("pick", normalized["kind"])
        self.assertIsNone(requested_show)
        self.assertNotIn("ambient_dir", normalized)


class PickCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.holidays = self.root / "holidays_catalog.json"
        self.ambient = self.root / "ambient_catalog.json"

        uploader.HOLIDAY_CATALOG_PATH = self.holidays
        uploader.AMBIENT_CATALOG_PATH = self.ambient

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_get_catalog_for_local_pick_holiday(self):
        pick_file = Path("/media/frame_ai/holidays/christmas/day/scene.jpg")
        catalog_path, catalog_key = uploader.get_catalog_for_local_pick(pick_file)
        self.assertEqual(self.holidays, catalog_path)
        self.assertEqual("christmas/day/scene.jpg", catalog_key)

    def test_get_catalog_for_local_pick_ambient(self):
        pick_file = Path("/media/frame_ai/ambient/winter/night/snow.png")
        catalog_path, catalog_key = uploader.get_catalog_for_local_pick(pick_file)
        self.assertEqual(self.ambient, catalog_path)
        self.assertEqual("winter/night/snow.png", catalog_key)

    def test_update_and_lookup_catalog_content_id(self):
        uploader.update_catalog_content_id(self.holidays, "christmas/day/scene.jpg", "MY_F123")
        cached = uploader.lookup_catalog_content_id(self.holidays, "christmas/day/scene.jpg")
        self.assertEqual("MY_F123", cached)

    def test_lookup_catalog_content_id_returns_none_for_missing(self):
        cached = uploader.lookup_catalog_content_id(self.ambient, "winter/night/missing.jpg")
        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
