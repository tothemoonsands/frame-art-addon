import unittest
from unittest import mock
import sys
import types
from pathlib import Path

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


class _FakeImage:
    def __init__(self, width=3840, height=2160):
        self.width = width
        self.height = height

    def save(self, out, format=None, optimize=None, compress_level=None):
        del format, optimize, compress_level
        out.write(b"img")


class CoverArtItunesTests(unittest.TestCase):
    def test_detects_openai_org_verification_error(self):
        err = ValueError(
            "OpenAI edits failed: 403 request_id=req_123 body={\"error\":{\"message\":\"Your organization must "
            "be verified to use the model `gpt-image-2`. Please complete API Organization Verification.\"}}"
        )

        self.assertTrue(cover_art.is_openai_org_verification_error(err))

    def test_ignores_non_verification_openai_error(self):
        err = ValueError("OpenAI edits failed: 500 request_id=req_123 body={\"error\":{\"message\":\"server error\"}}")

        self.assertFalse(cover_art.is_openai_org_verification_error(err))

    def test_verification_fallback_retries_with_gpt_image_1_5(self):
        verify_err = ValueError(
            "OpenAI edits failed: 403 request_id=req_123 body={\"error\":{\"message\":\"Your organization must "
            "be verified to use the model `gpt-image-2`. Please complete API Organization Verification.\"}}"
        )
        request_models = []

        def fake_request(**kwargs):
            request_models.append(kwargs["openai_model"])
            if len(request_models) == 1:
                raise verify_err
            return b"generated", "req_retry", "gpt-image-1.5"

        with (
            mock.patch("frame_art_uploader_ai.cover_art.build_reference_canvas_from_album", return_value=b"canvas"),
            mock.patch("frame_art_uploader_ai.cover_art._request_openai_reference_background", side_effect=fake_request),
            mock.patch("frame_art_uploader_ai.cover_art.ha_edit_to_frame", return_value=_FakeImage()),
            mock.patch("frame_art_uploader_ai.cover_art.composite_album", return_value=_FakeImage()),
        ):
            _, _, request_id, model_used = cover_art.generate_reference_frame_from_album(
                source_album_path=Path("album.png"),
                openai_api_key="key",
                openai_model="gpt-image-2",
                timeout_s=1,
            )

        self.assertEqual(["gpt-image-2", "gpt-image-1.5"], request_models)
        self.assertEqual("req_retry", request_id)
        self.assertEqual("gpt-image-1.5", model_used)

    def test_verification_fallback_does_not_retry_when_model_already_1_5(self):
        verify_err = ValueError(
            "OpenAI edits failed: 403 request_id=req_123 body={\"error\":{\"message\":\"Your organization must "
            "be verified to use the model `gpt-image-1.5`. Please complete API Organization Verification.\"}}"
        )

        with (
            mock.patch("frame_art_uploader_ai.cover_art.build_reference_canvas_from_album", return_value=b"canvas"),
            mock.patch(
                "frame_art_uploader_ai.cover_art._request_openai_reference_background",
                side_effect=verify_err,
            ) as mock_request,
        ):
            with self.assertRaises(ValueError):
                cover_art.generate_reference_frame_from_album(
                    source_album_path=Path("album.png"),
                    openai_api_key="key",
                    openai_model="gpt-image-1.5",
                    timeout_s=1,
                )

        self.assertEqual(1, mock_request.call_count)

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
