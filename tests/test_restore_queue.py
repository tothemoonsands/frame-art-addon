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
    cover.BACKGROUND_DIR = Path(".")
    cover.COMPRESSED_DIR = Path(".")
    cover.WIDESCREEN_DIR = Path(".")
    cover.JPEG_MAX_BYTES = 4 * 1024 * 1024
    cover.download_artwork = lambda *a, **k: None
    cover.compress_png_path_to_jpeg_max_bytes = lambda *a, **k: (True, 0)
    cover.ensure_dirs = lambda *a, **k: None
    cover.itunes_lookup = lambda *a, **k: {}
    cover.itunes_search = lambda *a, **k: {}
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


class MusicAssociationLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.assoc = self.root / "music_associations.json"
        self.catalog = self.root / "music_catalog.json"
        self.index = self.root / "index.json"
        uploader.MUSIC_ASSOCIATIONS_PATH = self.assoc
        uploader.MUSIC_CATALOG_PATH = self.catalog
        uploader.MUSIC_INDEX_PATHS = [self.index]

    def tearDown(self) -> None:
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
        self.assertEqual("catalog_index_text_key", matched.get("match_source"))

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


if __name__ == "__main__":
    unittest.main()
