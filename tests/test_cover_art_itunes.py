import unittest
from unittest import mock
import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *a, **k: None
    requests_stub.post = lambda *a, **k: None
    sys.modules["requests"] = requests_stub
if "PIL" not in sys.modules:
    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = object()
    pil_mod.ImageDraw = object()
    pil_mod.ImageEnhance = object()
    pil_mod.ImageFilter = object()
    pil_mod.ImageOps = object()
    sys.modules["PIL"] = pil_mod

from frame_art_uploader_ai import cover_art


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = "{}"

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class CoverArtItunesTests(unittest.TestCase):
    def test_normalize_key_collab_variants_share_cache_key(self):
        a = cover_art.normalize_key(None, "Loyle Carner", "Yesterday's Gone")
        b = cover_art.normalize_key(None, "Loyle Carner & Tom Misch", "Yesterday's Gone")
        c = cover_art.normalize_key(None, "Loyle Carner, Rebel Kleff & Kiko Bun", "Yesterday's Gone")
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    @mock.patch("frame_art_uploader_ai.cover_art.requests.get")
    def test_itunes_search_rejects_weak_match(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "collectionId": 1,
                        "collectionName": "Completely Different Album",
                        "artistName": "Unrelated Artist",
                    }
                ]
            }
        )

        got = cover_art.itunes_search("Loyle Carner", "Not Waving, But Drowning", timeout_s=1)
        self.assertEqual({}, got)

    @mock.patch("frame_art_uploader_ai.cover_art.requests.get")
    def test_itunes_search_can_match_collab_artist_variants(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "collectionId": 2,
                        "collectionName": "Not Waving, But Drowning",
                        "artistName": "Loyle Carner",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    },
                    {
                        "collectionId": 3,
                        "collectionName": "Hopefully",
                        "artistName": "Loyle Carner",
                    },
                ]
            }
        )

        got = cover_art.itunes_search("Loyle Carner & Tom Misch", "Not Waving, But Drowning", timeout_s=1)
        self.assertEqual(2, got.get("collectionId"))

    @mock.patch("frame_art_uploader_ai.cover_art.requests.get")
    def test_itunes_search_prefers_exact_album_over_edition_noise(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "collectionId": 20,
                        "collectionName": "Nothing New Under the Sun (Deluxe)",
                        "artistName": "Frankie Stew and Harvey Gunn",
                    },
                    {
                        "collectionId": 21,
                        "collectionName": "Nothing New Under the Sun",
                        "artistName": "Frankie Stew and Harvey Gunn",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    },
                ]
            }
        )

        got = cover_art.itunes_search("Frankie Stew and Harvey Gunn", "Nothing New Under the Sun", timeout_s=1)
        self.assertEqual(21, got.get("collectionId"))

    @mock.patch("frame_art_uploader_ai.cover_art.requests.get")
    def test_itunes_search_uses_us_country_and_album_first_term(self, mock_get):
        mock_get.return_value = _FakeResponse({"results": []})

        cover_art.itunes_search("Frankie Stew and Harvey Gunn", "Nothing New Under the Sun", timeout_s=1)

        first_url = mock_get.call_args_list[0].args[0]
        self.assertIn("country=us", first_url)
        self.assertIn("term=Nothing+New+Under+the+Sun+Frankie+Stew+and+Harvey+Gunn", first_url)

    @mock.patch("frame_art_uploader_ai.cover_art.requests.get")
    def test_itunes_track_search_picks_best_track(self, mock_get):
        mock_get.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "trackId": 11,
                        "trackName": "Not Waving, But Drowning",
                        "artistName": "Loyle Carner",
                        "collectionId": 200,
                        "collectionName": "Yesterday's Gone",
                        "artworkUrl100": "https://example.com/100x100bb.jpg",
                    },
                    {
                        "trackId": 12,
                        "trackName": "Another Song",
                        "artistName": "Someone Else",
                    },
                ]
            }
        )

        got = cover_art.itunes_track_search("Loyle Carner", "Not Waving, But Drowning", timeout_s=1)
        self.assertEqual(11, got.get("trackId"))
        self.assertEqual(200, got.get("collectionId"))


if __name__ == "__main__":
    unittest.main()
