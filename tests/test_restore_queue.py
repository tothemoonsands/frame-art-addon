import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
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
    cover.BACKGROUND_DIR = Path(".")
    cover.COMPRESSED_DIR = Path(".")
    cover.WIDESCREEN_DIR = Path(".")
    cover.JPEG_MAX_BYTES = 4 * 1024 * 1024
    cover.download_artwork = lambda *a, **k: None
    cover.compress_png_path_to_jpeg_max_bytes = lambda *a, **k: (True, 0)
    cover.ensure_dirs = lambda *a, **k: None
    cover.itunes_lookup = lambda *a, **k: {}
    cover.itunes_search = lambda *a, **k: {}
    cover.itunes_track_search = lambda *a, **k: {}
    cover.normalize_key = lambda *a, **k: "k"
    cover.generate_reference_frame_from_album = lambda *a, **k: (b"", b"", None, None)
    cover.generate_local_fallback_frame_from_album = lambda *a, **k: (b"", b"")
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
        self.music_overrides = self.root / "frame_art_music_overrides.json"
        self.music_catalog = self.root / "frame_art_music_catalog.json"

        uploader.RESTORE_REQUEST_PATH = str(self.inbox)
        uploader.RESTORE_QUEUE_DIR = self.queue
        uploader.WORKER_LOCK_PATH = self.lock
        uploader.MUSIC_OVERRIDES_PATH = self.music_overrides
        uploader.MUSIC_CATALOG_PATH = self.music_catalog

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

    def test_music_request_is_superseded_by_newer_music_request(self):
        self._write_inbox(
            {
                "kind": "cover_art_reference",
                "artist": "Old Artist",
                "album": "Old Album",
            }
        )
        first = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(first)

        self._write_inbox(
            {
                "kind": "cover_art_outpaint",
                "artist": "New Artist",
                "album": "New Album",
            }
        )
        second = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(second)

        self.assertTrue(uploader.is_superseded_music_request(first, "cover_art_reference_background"))

    def test_music_request_not_superseded_by_newer_pick(self):
        self._write_inbox(
            {
                "kind": "cover_art_reference",
                "artist": "Old Artist",
                "album": "Old Album",
            }
        )
        first = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(first)

        self._write_inbox({"kind": "pick", "phase": "night", "season": "winter"})
        second = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(second)

        self.assertFalse(uploader.is_superseded_music_request(first, "cover_art_reference_background"))

    def test_music_request_not_superseded_when_newer_music_show_false(self):
        self._write_inbox(
            {
                "kind": "cover_art_reference",
                "artist": "Old Artist",
                "album": "Old Album",
            }
        )
        first = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(first)

        self._write_inbox(
            {
                "kind": "cover_art_outpaint",
                "artist": "New Artist",
                "album": "New Album",
                "show": False,
            }
        )
        second = uploader.enqueue_restore_inbox_if_present()
        self.assertIsNotNone(second)

        self.assertFalse(uploader.is_superseded_music_request(first, "cover_art_reference_background"))

    def test_parse_music_feedback_payload(self):
        payload = {
            "kind": "music_feedback",
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "show": True,
            "action": "regen_now",
            "issue_type": "wrong_album_art",
            "artist": "Artist",
            "album": "Album",
            "track": "Track",
            "music_session_key": "session-1",
            "collection_id": "123",
            "current_content_id": "MY_F42",
            "cache_key": "itc_123",
            "candidate_catalog_key": "123__3840x2160.jpg",
            "notes": "bad background",
        }
        normalized, requested_show, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertTrue(requested_show)
        self.assertEqual("music_feedback", normalized["kind"])
        self.assertEqual("regen_now", normalized["action"])
        self.assertEqual("wrong_album_art", normalized["issue_type"])
        self.assertEqual(123, normalized["collection_id"])
        self.assertEqual("MY_F42", normalized["current_content_id"])

    def test_manual_override_preferred_for_lookup(self):
        ok = uploader.set_music_override_for_album(
            artist="The Artist",
            album="The Album",
            catalog_key="123__3840x2160.jpg",
            reason="manual",
        )
        self.assertTrue(ok)
        uploader.atomic_write_json(
            self.music_catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "123__3840x2160.jpg": {
                        "content_id": "MY_F123",
                        "updated_at": "",
                    }
                },
            },
        )
        match = uploader.lookup_music_association(
            {
                "artist": "The Artist",
                "album": "The Album",
            }
        )
        self.assertIsNotNone(match)
        self.assertEqual("manual_override", match.get("match_source"))
        self.assertEqual("MY_F123", match.get("content_id"))


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
        self.assertTrue(normalized["apply_deletions"])
        self.assertTrue(normalized["auto_queue_missing"])
        self.assertEqual(25, normalized["delete_limit"])

    def test_parse_dispatch_holiday_and_music_seed_defaults(self):
        holiday, _, err_holiday = uploader.parse_restore_request_payload({"kind": "holiday_seed"})
        music, _, err_music = uploader.parse_restore_request_payload({"kind": "music_seed"})
        self.assertIsNone(err_holiday)
        self.assertIsNone(err_music)
        self.assertEqual("holiday_seed", holiday["kind"])
        self.assertEqual("music_seed", music["kind"])
        self.assertTrue(holiday["apply_deletions"])
        self.assertTrue(music["auto_queue_missing"])
        self.assertEqual(25, holiday["delete_limit"])
        self.assertEqual(25, music["delete_limit"])

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

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_pending_delete_is_processed_and_marked_deleted(self, mocked_swap, mocked_delete):
        (self.ambient / "keep.jpg").write_bytes(b"k")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "old.jpg": {
                        "content_id": "MY_F111",
                        "state": "pending_delete",
                    }
                },
            },
        )
        mocked_swap.return_value = (True, False, None)
        mocked_delete.return_value = {"ok": True, "verified": True, "error": None}

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": False,
            },
            requested_at="",
        )

        self.assertEqual(1, status["deletion_candidates"])
        self.assertEqual(1, status["deletion_processed"])
        data = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("deleted", data["entries"]["old.jpg"]["state"])

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_current_display_swap_failure_leaves_pending_delete(self, mocked_swap, mocked_delete):
        (self.ambient / "keep.jpg").write_bytes(b"k")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "old.jpg": {
                        "content_id": "MY_F111",
                        "state": "pending_delete",
                    }
                },
            },
        )
        mocked_swap.return_value = (False, False, "swap_failed_for_current:MY_F111")

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "force_reupload": False,
            },
            requested_at="",
        )

        self.assertEqual(1, status["deletion_failed"])
        self.assertEqual(1, status["deletion_skipped_currently_displayed"])
        self.assertEqual(0, mocked_delete.call_count)
        data = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("pending_delete", data["entries"]["old.jpg"]["state"])

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_missing_file_is_auto_queued_and_deleted(self, mocked_swap, mocked_delete):
        (self.ambient / "keep.jpg").write_bytes(b"k")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "missing.jpg": {
                        "content_id": "MY_F404",
                        "state": "active",
                    }
                },
            },
        )
        mocked_swap.return_value = (True, False, None)
        mocked_delete.return_value = {"ok": True, "verified": True, "error": None}

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
                "auto_queue_missing": True,
            },
            requested_at="",
        )

        self.assertEqual(1, status["auto_queued_missing_count"])
        self.assertEqual(1, status["deletion_processed"])
        data = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("deleted", data["entries"]["missing.jpg"]["state"])

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_delete_failure_keeps_pending_and_reports_status_fields(self, mocked_swap, mocked_delete):
        (self.ambient / "keep.jpg").write_bytes(b"k")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "old.jpg": {
                        "content_id": "MY_FAIL",
                        "state": "pending_delete",
                    }
                },
            },
        )
        mocked_swap.return_value = (True, False, None)
        mocked_delete.return_value = {"ok": False, "verified": False, "error": "delete_unverified:MY_FAIL"}

        status = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={
                "kind": "ambient_seed",
                "ambient_dir": str(self.ambient),
                "catalog_path": str(self.catalog),
            },
            requested_at="",
        )

        self.assertIn("deletion_candidates", status)
        self.assertIn("deletion_processed", status)
        self.assertIn("deletion_failed", status)
        self.assertIn("deletion_skipped_currently_displayed", status)
        self.assertIn("deletion_swap_fallback_used", status)
        self.assertIn("deletion_errors", status)
        self.assertEqual(1, status["deletion_failed"])
        data = json.loads(self.catalog.read_text(encoding="utf-8"))
        self.assertEqual("pending_delete", data["entries"]["old.jpg"]["state"])

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_delete_idempotency_second_run_no_duplicate_failures(self, mocked_swap, mocked_delete):
        (self.ambient / "keep.jpg").write_bytes(b"k")
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "old.jpg": {
                        "content_id": "MY_F111",
                        "state": "pending_delete",
                    }
                },
            },
        )
        mocked_swap.return_value = (True, False, None)
        mocked_delete.return_value = {"ok": True, "verified": True, "error": None}

        first = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={"kind": "ambient_seed", "ambient_dir": str(self.ambient), "catalog_path": str(self.catalog)},
            requested_at="",
        )
        second = uploader.handle_ambient_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={"kind": "ambient_seed", "ambient_dir": str(self.ambient), "catalog_path": str(self.catalog)},
            requested_at="",
        )

        self.assertEqual(1, first["deletion_processed"])
        self.assertEqual(0, second["deletion_processed"])
        self.assertEqual(0, second["deletion_failed"])


class DeleteApiPathTests(unittest.TestCase):
    def test_delete_prefers_delete_list_with_list_payload(self):
        art = mock.Mock()
        art.delete_list = mock.Mock()
        art.available.return_value = []

        result = uploader.delete_art_content_id(art, "MY_F999")

        self.assertTrue(result["ok"])
        art.delete_list.assert_called_once_with(["MY_F999"])

    def test_delete_falls_back_to_delete_method(self):
        art = mock.Mock()
        art.delete_list = None
        art.delete = mock.Mock()
        art.available.return_value = []

        result = uploader.delete_art_content_id(art, "MY_F123")

        self.assertTrue(result["ok"])
        art.delete.assert_called_once_with("MY_F123")


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


class MusicSeedDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.music_dir = self.root / "music_seed"
        self.catalog = self.root / "music_catalog.json"
        self.assoc = self.root / "music_assoc.json"
        self.index = self.root / "music_index.json"
        self.widescreen = self.root / "widescreen"
        self.compressed = self.root / "compressed"
        self.background = self.root / "background"
        self.source = self.root / "source"
        self.music_dir.mkdir(parents=True)
        self.widescreen.mkdir(parents=True)
        self.compressed.mkdir(parents=True)
        self.background.mkdir(parents=True)
        self.source.mkdir(parents=True)
        (self.music_dir / "keep.jpg").write_bytes(b"keep")

        self.old_music_catalog = uploader.MUSIC_CATALOG_PATH
        self.old_music_assoc = uploader.MUSIC_ASSOCIATIONS_PATH
        self.old_index_paths = uploader.MUSIC_INDEX_PATHS
        self.old_widescreen = uploader.WIDESCREEN_DIR
        self.old_compressed = uploader.COMPRESSED_DIR
        self.old_background = uploader.BACKGROUND_DIR
        self.old_source = uploader.SOURCE_DIR
        uploader.MUSIC_CATALOG_PATH = self.catalog
        uploader.MUSIC_ASSOCIATIONS_PATH = self.assoc
        uploader.MUSIC_INDEX_PATHS = [self.index]
        uploader.WIDESCREEN_DIR = self.widescreen
        uploader.COMPRESSED_DIR = self.compressed
        uploader.BACKGROUND_DIR = self.background
        uploader.SOURCE_DIR = self.source

    def tearDown(self) -> None:
        uploader.MUSIC_CATALOG_PATH = self.old_music_catalog
        uploader.MUSIC_ASSOCIATIONS_PATH = self.old_music_assoc
        uploader.MUSIC_INDEX_PATHS = self.old_index_paths
        uploader.WIDESCREEN_DIR = self.old_widescreen
        uploader.COMPRESSED_DIR = self.old_compressed
        uploader.BACKGROUND_DIR = self.old_background
        uploader.SOURCE_DIR = self.old_source
        self.tmp.cleanup()

    @mock.patch("frame_art_uploader_ai.uploader.delete_art_content_id")
    @mock.patch("frame_art_uploader_ai.uploader.maybe_swap_current_art")
    def test_music_pending_delete_cleans_graph_and_files(self, mocked_swap, mocked_delete):
        key = "123__3840x2160.jpg"
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    key: {"content_id": "MY_F123", "state": "pending_delete"},
                },
            },
        )
        uploader.atomic_write_json(
            self.assoc,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "cache::123": {"catalog_key": key, "content_id": "MY_F123"},
                    "album_norm::x": {"catalog_key": "other.jpg", "content_id": "MY_OTHER"},
                },
            },
        )
        uploader.atomic_write_json(
            self.index,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "123": {"content_id": "MY_F123", "compressed_output_path": str(self.compressed / key)},
                    "keep": {"content_id": "MY_KEEP"},
                },
            },
        )
        (self.compressed / key).write_bytes(b"x")
        (self.widescreen / key).write_bytes(b"x")
        (self.background / "123__3840x2160__background.png").write_bytes(b"x")
        (self.source / "123.jpg").write_bytes(b"x")

        mocked_swap.return_value = (True, False, None)
        mocked_delete.return_value = {"ok": True, "verified": True, "error": None}

        status = uploader.handle_music_seed_restore(
            tv_ip="127.0.0.1",
            art=mock.Mock(),
            restore_payload={
                "kind": "music_seed",
                "music_dir": str(self.music_dir),
                "catalog_path": str(self.catalog),
            },
            requested_at="",
        )

        self.assertEqual(1, status["deletion_processed"])
        assoc = json.loads(self.assoc.read_text(encoding="utf-8"))
        self.assertNotIn("cache::123", assoc["entries"])
        index = json.loads(self.index.read_text(encoding="utf-8"))
        self.assertNotIn("123", index["entries"])
        self.assertFalse((self.compressed / key).exists())
        self.assertFalse((self.widescreen / key).exists())
        self.assertFalse((self.background / "123__3840x2160__background.png").exists())
        self.assertFalse((self.source / "123.jpg").exists())

class CoverArtPayloadNormalizationTests(unittest.TestCase):
    def test_non_shazam_source_clears_shazam_key(self):
        payload = {
            "kind": "cover_art_reference",
            "key_source": "sonos",
            "shazam_key": "album:yes|stale",
            "artist": "Michael Kiwanuka",
            "album": "Home Again",
        }
        normalized, _, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertEqual("cover_art_reference_background", normalized["kind"])
        self.assertEqual("sonos", normalized["key_source"])
        self.assertEqual("", normalized["shazam_key"])

    def test_shazam_source_keeps_shazam_key(self):
        payload = {
            "kind": "cover_art_reference",
            "key_source": "shazam",
            "shazam_key": "album:yes|home-again",
            "artist": "Michael Kiwanuka",
            "album": "Home Again",
        }
        normalized, _, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertEqual("shazam", normalized["key_source"])
        self.assertEqual("album:yes|home-again", normalized["shazam_key"])

    def test_boolean_like_key_source_with_shazam_key_is_inferred_as_shazam(self):
        payload = {
            "kind": "cover_art_reference",
            "key_source": "true",
            "shazam_key": "album:childish gambino|\"awaken, my love!\"",
            "artist": "love!",
            "album": "album",
            "track": "Childish Gambino",
            "music_session_key": "album:childish gambino|awaken,",
        }
        normalized, _, err = uploader.parse_restore_request_payload(payload)
        self.assertIsNone(err)
        self.assertEqual("shazam", normalized["key_source"])
        self.assertEqual("album:childish gambino|\"awaken, my love!\"", normalized["shazam_key"])
        self.assertEqual("childish gambino", normalized["artist"].lower())
        self.assertEqual("awaken, my love!", normalized["album"].lower())

    def test_shazam_key_parsing_recovers_quoted_album(self):
        artist, album = uploader.parse_shazam_album_match_key('album:childish gambino|"awaken, my love!"')
        self.assertEqual("childish gambino", artist.lower())
        self.assertEqual("awaken, my love!", album.lower())

    def test_normalized_album_association_collab_variants_share_key(self):
        base = uploader.normalized_album_association("Loyle Carner", "Yesterday's Gone")
        collab = uploader.normalized_album_association("Loyle Carner & Tom Misch", "Yesterday's Gone")
        multi = uploader.normalized_album_association(
            "Loyle Carner, Rebel Kleff & Kiko Bun",
            "Yesterday's Gone",
        )
        self.assertEqual(base, collab)
        self.assertEqual(base, multi)


class MusicAssociationLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.assoc = self.root / "music_associations.json"
        self.catalog = self.root / "music_catalog.json"
        self.index = self.root / "index.json"
        self.old_compressed_dir = uploader.COMPRESSED_DIR
        self.old_widescreen_dir = uploader.WIDESCREEN_DIR
        uploader.COMPRESSED_DIR = self.root / "widescreen-compressed"
        uploader.WIDESCREEN_DIR = self.root / "widescreen"
        uploader.COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
        uploader.WIDESCREEN_DIR.mkdir(parents=True, exist_ok=True)
        uploader.MUSIC_ASSOCIATIONS_PATH = self.assoc
        uploader.MUSIC_CATALOG_PATH = self.catalog
        uploader.MUSIC_INDEX_PATHS = [self.index]

    def tearDown(self) -> None:
        uploader.COMPRESSED_DIR = self.old_compressed_dir
        uploader.WIDESCREEN_DIR = self.old_widescreen_dir
        self.tmp.cleanup()

    def test_lookup_by_normalized_album_key(self):
        uploader.update_music_association(
            {
                "artist": "Michael Kiwanuka",
                "album": "Home Again",
                "track": "Tell Me a Tale",
                "key_source": "sonos",
            },
            cache_key="k1",
            catalog_key="k1__3840x2160.jpg",
            content_id="MY_F123",
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "michael   kiwanuka",
                "album": "HOME AGAIN",
            }
        )
        self.assertIsInstance(matched, dict)
        self.assertEqual("k1__3840x2160.jpg", matched.get("catalog_key"))

    def test_lookup_legacy_album_key_without_album_norm(self):
        uploader.atomic_write_json(
            self.assoc,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "album::Michael Kiwanuka — Home Again": {
                        "catalog_key": "legacy__3840x2160.jpg",
                        "content_id": "MY_F999",
                    }
                },
            },
        )
        matched = uploader.lookup_music_association(
            {
                "artist": "michael kiwanuka",
                "album": "home again",
            }
        )
        self.assertIsInstance(matched, dict)
        self.assertEqual("legacy__3840x2160.jpg", matched.get("catalog_key"))

    def test_fuzzy_lookup_from_association_record(self):
        uploader.update_music_association(
            {
                "artist": "Milt Jackson",
                "album": "Sunflower (40th Anniversary Edition)",
                "key_source": "sonos",
            },
            cache_key="sunflower_k1",
            catalog_key="aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9__3840x2160.jpg",
            content_id="MY_F1001",
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "Milt Jackson",
                "album": "Sunflower 40th Anniversary",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertEqual("MY_F1001", matched.get("content_id"))
        self.assertEqual("association_fuzzy", matched.get("match_source"))

    def test_fuzzy_lookup_from_catalog_filename(self):
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9__3840x2160.jpg": {
                        "content_id": "MY_F2002",
                        "updated_at": "",
                    }
                },
            },
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "Milt Jackson",
                "album": "Sunflower 40th Anniversary Edition",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertEqual(
            "aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9__3840x2160.jpg",
            matched.get("catalog_key"),
        )
        self.assertEqual("catalog_filename_fuzzy", matched.get("match_source"))

    def test_fuzzy_lookup_from_index_text_key_for_numeric_catalog(self):
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "1440795324__3840x2160.jpg": {
                        "content_id": "MY_F1226",
                        "updated_at": "",
                    }
                },
            },
        )
        uploader.atomic_write_json(
            self.index,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "1440795324": {
                        "text_key": "michael kiwanuka — home again",
                        "compressed_output_path": "/tmp/out/widescreen-compressed/1440795324__3840x2160.jpg",
                    }
                },
            },
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "Michael Kiwanuka",
                "album": "Home Again",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertEqual("1440795324__3840x2160.jpg", matched.get("catalog_key"))
        self.assertEqual(1440795324, matched.get("collection_id"))
        self.assertEqual("index_text_key_exact", matched.get("match_source"))

    def test_exact_index_text_key_match_finds_existing_numeric_cache_file(self):
        expected_file = uploader.COMPRESSED_DIR / "135132797__3840x2160.jpg"
        expected_file.write_bytes(b"jpg")
        uploader.atomic_write_json(
            self.index,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "135132797": {
                        "text_key": "Tom Misch — Six Songs EP",
                        "compressed_output_path": str(expected_file),
                        "content_id": "MY_F555",
                    }
                },
            },
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "tom misch",
                "album": "six songs ep",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertEqual("135132797__3840x2160.jpg", matched.get("catalog_key"))
        self.assertEqual(135132797, matched.get("collection_id"))
        self.assertEqual("index_text_key_exact", matched.get("match_source"))
        self.assertEqual("MY_F555", matched.get("content_id"))

    def test_exact_index_match_allows_and_token_difference(self):
        expected_file = uploader.COMPRESSED_DIR / "1497226866__3840x2160.jpg"
        expected_file.write_bytes(b"jpg")
        uploader.atomic_write_json(
            self.index,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "1497226866": {
                        "text_key": "tom misch and yussef dayes — what kinda music",
                        "compressed_output_path": str(expected_file),
                        "content_id": "MY_F1362",
                    }
                },
            },
        )

        matched = uploader.lookup_music_association(
            {
                "artist": "tom misch yussef dayes",
                "album": "what kinda music",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertEqual("1497226866__3840x2160.jpg", matched.get("catalog_key"))
        self.assertEqual("index_text_key_exact", matched.get("match_source"))
        self.assertEqual("MY_F1362", matched.get("content_id"))

    def test_fuzzy_lookup_requires_clear_winner_margin(self):
        uploader.atomic_write_json(
            self.assoc,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "album::Tom Misch — Six Songs EP": {
                        "cache_key": "aa_tom-misch-six-songs-ep_a1b2c3d4",
                        "catalog_key": "aa_tom-misch-six-songs-ep_a1b2c3d4__3840x2160.jpg",
                        "content_id": "MY_F1",
                        "artist": "Tom Misch",
                        "album": "Six Songs EP",
                    },
                    "album::Tom Misch — Six Songs EP (Alt)": {
                        "cache_key": "aa_tom-misch-six-songs-ep_z9y8x7w6",
                        "catalog_key": "aa_tom-misch-six-songs-ep_z9y8x7w6__3840x2160.jpg",
                        "content_id": "MY_F2",
                        "artist": "Tom Misch",
                        "album": "Six Songs EP",
                    },
                },
            },
        )

        matched = uploader.lookup_music_association_fuzzy(
            {
                "artist": "Tom Misch",
                "album": "Six Songs EP",
            }
        )

        self.assertIsInstance(matched, dict)
        self.assertTrue(matched.get("generation_blocked"))

    def test_lookup_prefers_collection_id_numeric_catalog_match(self):
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "1497226866__3840x2160.jpg": {
                        "content_id": "MY_F1362",
                        "updated_at": "",
                    }
                },
            },
        )
        matched = uploader.lookup_music_association(
            {
                "collection_id": 1497226866,
                "artist": "tom misch yussef dayes",
                "album": "what kinda music",
            }
        )
        self.assertIsInstance(matched, dict)
        self.assertEqual("collection_id_catalog_exact", matched.get("match_source"))
        self.assertEqual("1497226866__3840x2160.jpg", matched.get("catalog_key"))
        self.assertEqual("MY_F1362", matched.get("content_id"))

    def test_fuzzy_ambiguous_returns_generation_block(self):
        uploader.atomic_write_json(
            self.catalog,
            {
                "version": 1,
                "updated_at": "",
                "entries": {
                    "aa_tom-misch-yussef-dayes-what-kinda-music_11111111__3840x2160.jpg": {
                        "content_id": "MY_F1",
                        "updated_at": "",
                    },
                    "aa_tom-misch-yussef-dayes-what-kinda-music_22222222__3840x2160.jpg": {
                        "content_id": "MY_F2",
                        "updated_at": "",
                    },
                },
            },
        )

        matched = uploader.lookup_music_association_fuzzy(
            {
                "artist": "tom misch and yussef dayes",
                "album": "what kinda music",
            }
        )
        self.assertIsInstance(matched, dict)
        self.assertTrue(matched.get("generation_blocked"))
        self.assertEqual("fuzzy_ambiguous_blocked", matched.get("match_source"))
        self.assertIsInstance(matched.get("match_candidates"), list)

    def test_update_music_index_entry_uses_collection_id_key(self):
        wide = self.root / "1440795324__3840x2160.jpg"
        wide.write_bytes(b"x")
        uploader.update_music_index_entry(
            artist="Michael Kiwanuka",
            album="Home Again",
            collection_id=1440795324,
            catalog_key="1440795324__3840x2160.jpg",
            cache_key="itc_1440795324",
            content_id="MY_F1226",
            wide_path=wide,
            compressed_path=wide,
            request_id="req_123",
        )
        index = json.loads(self.index.read_text(encoding="utf-8"))
        entry = index["entries"]["1440795324"]
        self.assertEqual("michael kiwanuka — home again", entry["text_key"])
        self.assertEqual("ok", entry["status"])
        self.assertEqual("MY_F1226", entry["content_id"])
        self.assertEqual("reference_background_nomask", entry["prompt_variant"])
        self.assertEqual("req_123", entry["request_id"])

    def test_update_music_index_entry_uses_cache_stem_key_without_collection_id(self):
        wide = self.root / "aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9__3840x2160.jpg"
        wide.write_bytes(b"x")
        uploader.update_music_index_entry(
            artist="Milt Jackson",
            album="Sunflower 40th Anniversary Edition",
            collection_id=None,
            catalog_key=wide.name,
            cache_key="aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9",
            content_id="MY_F2002",
            wide_path=wide,
            compressed_path=wide,
            request_id=None,
        )
        index = json.loads(self.index.read_text(encoding="utf-8"))
        entry_key = "aa_milt-jackson-sunflower-40th-anniversary-edition_8d74a5e9"
        self.assertIn(entry_key, index["entries"])
        self.assertEqual("milt jackson — sunflower 40th anniversary edition", index["entries"][entry_key]["text_key"])
        self.assertEqual("MY_F2002", index["entries"][entry_key]["content_id"])

    def test_update_music_index_entry_skips_invalid_json_without_overwrite(self):
        self.index.write_text("{not-json", encoding="utf-8")
        original = self.index.read_text(encoding="utf-8")
        wide = self.root / "1440795324__3840x2160.jpg"
        wide.write_bytes(b"x")
        uploader.update_music_index_entry(
            artist="Michael Kiwanuka",
            album="Home Again",
            collection_id=1440795324,
            catalog_key="1440795324__3840x2160.jpg",
            cache_key="itc_1440795324",
            content_id="MY_F1226",
            wide_path=wide,
            compressed_path=wide,
            request_id="req_123",
        )
        self.assertEqual(original, self.index.read_text(encoding="utf-8"))

    def test_compact_music_association_entries_trims_duplicate_aliases(self):
        record = {
            "cache_key": "itc_1497226866",
            "catalog_key": "1497226866__3840x2160.jpg",
            "content_id": "MY_F1362",
            "key_source": "sonos",
            "artist": "Tom Misch & Yussef Dayes",
            "album": "What Kinda Music",
            "collection_id": 1497226866,
            "verified": True,
            "source_quality": "trusted_cache",
            "updated_at": "2026-02-10T12:27:37.002937+00:00",
        }
        entries = {
            "session::What Kinda Music": dict(record),
            "album::Tom Misch & Yussef Dayes — What Kinda Music": dict(record),
            "album_norm::tom misch yussef dayes what kinda music": dict(record),
            "album_loose::tom misch yussef dayes what kinda music": dict(record),
            "cache::itc_1497226866": dict(record),
        }

        compacted, stats = uploader.compact_music_association_entries(entries, session_ttl_days=30)
        self.assertLess(stats["output_entries"], stats["input_entries"])
        self.assertIn("album_norm::tom misch what kinda music", compacted)
        self.assertIn("cache::itc_1497226866", compacted)
        self.assertIn("session::What Kinda Music", compacted)
        self.assertNotIn("album::Tom Misch & Yussef Dayes — What Kinda Music", compacted)
        self.assertNotIn("album_loose::tom misch yussef dayes what kinda music", compacted)

    def test_update_music_association_writes_compact_aliases(self):
        uploader.update_music_association(
            {
                "music_session_key": "What Kinda Music",
                "artist": "Tom Misch & Yussef Dayes",
                "album": "What Kinda Music",
                "key_source": "sonos",
            },
            cache_key="itc_1497226866",
            catalog_key="1497226866__3840x2160.jpg",
            content_id="MY_F1362",
            verified=True,
            source_quality="trusted_cache",
        )

        assoc = json.loads(self.assoc.read_text(encoding="utf-8"))
        entries = assoc["entries"]
        self.assertIn("album_norm::tom misch what kinda music", entries)
        self.assertIn("cache::itc_1497226866", entries)
        self.assertIn("session::What Kinda Music", entries)
        self.assertNotIn("album::Tom Misch & Yussef Dayes — What Kinda Music", entries)
        self.assertNotIn("album_loose::tom misch yussef dayes what kinda music", entries)


if __name__ == "__main__":
    unittest.main()
