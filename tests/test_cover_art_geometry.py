import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageChops

from frame_art_uploader_ai import cover_art


class _OpenAIResponse:
    headers = {"x-request-id": "req_geometry"}
    status_code = 200
    text = "response"

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "gpt-image-test",
            "data": [{"b64_json": base64.b64encode(b"generated").decode("ascii")}],
        }


class _ContextResponse:
    headers = {"x-request-id": "req_context"}
    status_code = 200
    text = "response"

    def raise_for_status(self):
        return None

    def json(self):
        profile = {
            "identity": "Independent community radio in Telluride, Colorado",
            "cultural_context": "Volunteer-powered programming rooted in the San Juan Mountains",
            "visual_subject": "Telluride valley beneath steep alpine peaks",
            "visual_language": "Textured contemporary landscape painting",
            "composition": "A panoramic valley with layered mountain planes",
            "palette": ["alpine green", "mineral blue", "ochre"],
            "mood": ["independent", "regional", "expansive"],
            "avoid": ["radio logos", "microphones", "tourism poster text"],
            "rationale": "The station's place and community identity are more distinctive than generic radio imagery",
        }
        return {
            "model": "gpt-5-mini-test",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [{"type": "url", "url": "https://koto.org/about/"}],
                    },
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(profile)}],
                },
            ],
        }


class CoverArtGeometryTests(unittest.TestCase):
    def test_reference_canvas_uses_final_frame_cover_proportion(self):
        self.assertEqual(
            (461, 205, 1075, 819),
            cover_art.reference_cover_box(cover_art.HA_EDIT_WIDTH, cover_art.HA_EDIT_HEIGHT),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.png"
            source = Image.new("RGB", (307, 307), (17, 83, 149))
            source.putpixel((0, 0), (241, 37, 19))
            source.putpixel((306, 306), (29, 211, 73))
            source.save(source_path)

            canvas_bytes = cover_art.build_reference_canvas_from_album(str(source_path))

        with Image.open(BytesIO(canvas_bytes)) as canvas:
            canvas = canvas.convert("RGB")
            self.assertEqual((1536, 1024), canvas.size)

            left, top, right, bottom = (461, 205, 1075, 819)
            expected_cover = source.resize((614, 614), Image.Resampling.LANCZOS)
            actual_cover = canvas.crop((left, top, right, bottom))
            self.assertIsNone(ImageChops.difference(expected_cover, actual_cover).getbbox())

            outside_regions = (
                (0, 0, canvas.width, top),
                (0, bottom, canvas.width, canvas.height),
                (0, top, left, bottom),
                (right, top, canvas.width, bottom),
            )
            for box in outside_regions:
                self.assertIsNone(canvas.crop(box).getbbox(), box)

    def test_final_composite_keeps_centered_pixel_identical_1536_cover(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.png"
            source = Image.new(
                "RGB",
                (cover_art.FINAL_ALBUM_SIZE, cover_art.FINAL_ALBUM_SIZE),
                (23, 101, 179),
            )
            source.putpixel((0, 0), (251, 31, 47))
            source.putpixel(
                (cover_art.FINAL_ALBUM_SIZE - 1, cover_art.FINAL_ALBUM_SIZE - 1),
                (53, 229, 71),
            )
            source.save(source_path)

            background = Image.new(
                "RGB",
                (cover_art.FRAME_FINAL_WIDTH, cover_art.FRAME_FINAL_HEIGHT),
                (7, 11, 13),
            )
            final = cover_art.composite_album(background, source_path)

        self.assertEqual((3840, 2160), final.size)
        left = (cover_art.FRAME_FINAL_WIDTH - cover_art.FINAL_ALBUM_SIZE) // 2
        top = (cover_art.FRAME_FINAL_HEIGHT - cover_art.FINAL_ALBUM_SIZE) // 2
        actual_cover = final.crop(
            (left, top, left + cover_art.FINAL_ALBUM_SIZE, top + cover_art.FINAL_ALBUM_SIZE)
        )
        self.assertEqual((1536, 1536), actual_cover.size)
        self.assertIsNone(ImageChops.difference(source, actual_cover).getbbox())

    @mock.patch("frame_art_uploader_ai.cover_art.requests.post")
    def test_openai_multipart_uses_only_image_array_and_no_mask(self, mock_post):
        mock_post.return_value = _OpenAIResponse()

        generated, request_id, model_used = cover_art._request_openai_reference_background_once(
            input_canvas_png=b"reference-canvas",
            openai_api_key="key",
            openai_model="gpt-image-test",
            prompt="test prompt",
            timeout_s=1,
        )

        self.assertEqual(b"generated", generated)
        self.assertEqual("req_geometry", request_id)
        self.assertEqual("gpt-image-test", model_used)
        request_kwargs = mock_post.call_args.kwargs
        self.assertEqual(["image[]"], [key for key, _ in request_kwargs["files"]])
        self.assertEqual("1536x1024", request_kwargs["data"]["size"])
        self.assertNotIn("mask", request_kwargs["data"])
        self.assertNotIn("image", request_kwargs["data"])
        self.assertNotIn("image[]", request_kwargs["data"])

    def test_session_prompt_uses_collection_and_bounded_track_context(self):
        tracks = [
            {
                "media_title": "Haunted Pumpkin",
                "media_artist": "PBdR",
                "media_album_name": "Lofi Girl - Halloween 2023",
            },
            {"media_title": "Witching Hour", "media_artist": "Jam'addict"},
        ]
        prompt = cover_art.build_session_background_prompt(
            "playlist", "Halloween lofi", tracks
        )

        self.assertIn("playlist: Halloween lofi", prompt)
        self.assertIn("Haunted Pumpkin", prompt)
        self.assertIn("Witching Hour", prompt)
        self.assertIn("no album cover", prompt.lower())
        self.assertIn("no turntables", prompt.lower())
        self.assertIn("speakers", prompt.lower())
        self.assertIn("vinyl records", prompt.lower())

    def test_radio_session_prompt_avoids_literal_lifestyle_cliches(self):
        prompt = cover_art.build_session_background_prompt(
            "radio",
            "CH 17 - The Coffee House",
            [{"media_title": "Your Place At My Place", "media_artist": "Joshua Slone"}],
        )
        prompt_lower = prompt.lower()
        self.assertIn("identity label", prompt_lower)
        self.assertIn("never as a literal scene request", prompt_lower)
        self.assertIn("steaming mug", prompt_lower)
        self.assertIn("staged sofa", prompt_lower)
        self.assertIn("aspirational interior", prompt_lower)
        self.assertIn("painterly", prompt_lower)

    def test_session_prompt_includes_curated_profile(self):
        profile = cover_art.fallback_session_art_profile("radio", "KOTO Radio", [])
        prompt = cover_art.build_session_background_prompt(
            "radio", "KOTO Radio", [], art_profile=profile
        )
        self.assertIn("Telluride valley", prompt)
        self.assertIn("San Juan Mountains", prompt)
        self.assertIn("contemporary landscape painting", prompt)

    @mock.patch("frame_art_uploader_ai.cover_art.requests.post")
    def test_context_planner_uses_optional_web_search_and_structured_output(self, mock_post):
        mock_post.return_value = _ContextResponse()
        profile, metadata = cover_art.request_session_art_profile(
            listening_mode="radio",
            collection_name="KOTO Radio",
            context_tracks=[{"media_title": "Morning Music", "media_artist": "KOTO DJ"}],
            openai_api_key="key",
            context_model="gpt-5-mini-test",
            timeout_s=1,
            enable_web_search=True,
        )
        self.assertIn("Telluride", profile["identity"])
        self.assertEqual("req_context", metadata["request_id"])
        self.assertTrue(metadata["used_web_search"])
        self.assertEqual(["https://koto.org/about/"], metadata["sources"])
        request_json = mock_post.call_args.kwargs["json"]
        self.assertEqual([{"type": "web_search", "search_context_size": "medium"}], request_json["tools"])
        self.assertEqual("json_schema", request_json["text"]["format"]["type"])
        self.assertFalse(request_json["store"])

    @mock.patch("frame_art_uploader_ai.cover_art.request_session_art_profile")
    def test_resolved_context_profile_is_cached_per_collection(self, mock_request):
        planned = cover_art.fallback_session_art_profile("radio", "KOTO Radio", [])
        mock_request.return_value = (planned, {"request_id": "req_1", "sources": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "profiles.json"
            first, first_meta = cover_art.resolve_session_art_profile(
                listening_mode="radio",
                collection_name="KOTO Radio",
                context_tracks=[],
                openai_api_key="key",
                cache_path=cache_path,
            )
            second, second_meta = cover_art.resolve_session_art_profile(
                listening_mode="radio",
                collection_name="KOTO Radio",
                context_tracks=[],
                openai_api_key="key",
                cache_path=cache_path,
            )
        self.assertEqual(first, second)
        self.assertEqual("openai_context_planner", first_meta["source"])
        self.assertEqual("cache", second_meta["source"])
        self.assertEqual(1, mock_request.call_count)

    @mock.patch("frame_art_uploader_ai.cover_art.request_session_art_profile")
    def test_context_planner_failures_use_genre_aware_local_profile(self, mock_request):
        mock_request.side_effect = ValueError("planner unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            profile, metadata = cover_art.resolve_session_art_profile(
                listening_mode="radio",
                collection_name="CH 45 - Shade 45",
                context_tracks=[{"media_artist": "Core DJs", "media_title": "Friday Mix"}],
                openai_api_key="key",
                context_model="gpt-6-astra",
                cache_path=Path(temp_dir) / "profiles.json",
            )
        self.assertEqual("local_fallback", metadata["source"])
        self.assertIn("Urban mixed media", profile["visual_language"])
        self.assertEqual(3, mock_request.call_count)

    def test_local_session_fallback_is_full_frame_and_coverless(self):
        final_png, background_png = cover_art.generate_local_session_background(
            "playlist: Halloween lofi"
        )

        self.assertEqual(final_png, background_png)
        with Image.open(BytesIO(final_png)) as image:
            self.assertEqual((3840, 2160), image.size)


if __name__ == "__main__":
    unittest.main()
