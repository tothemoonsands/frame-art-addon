import base64
from io import BytesIO
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


if __name__ == "__main__":
    unittest.main()
