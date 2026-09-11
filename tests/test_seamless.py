import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from frame_art_uploader_ai import cover_art, seamless


class SeamlessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.source=Path(self.temp.name)/'cover.png'
        Image.new('RGB',(1536,1536),(25,110,200)).save(self.source)

    def tearDown(self):self.temp.cleanup()

    def test_original_finish_matches_legacy_library_and_preserves_edges(self):
        background=Image.new('RGB',(3840,2160),(180,190,200))
        actual=seamless.composite(background,self.source)
        alpha=Image.new('RGBA',background.size,(0,0,0,0))
        ImageDraw.Draw(alpha).rectangle((1152,328,2688,1864),fill=(0,0,0,88))
        expected=Image.alpha_composite(background.convert('RGBA'),alpha.filter(ImageFilter.GaussianBlur(26)))
        with Image.open(self.source) as cover:expected.paste(cover,(1152,312))
        self.assertIsNone(ImageChops.difference(actual,expected.convert('RGB')).getbbox())
        self.assertEqual((25,110,200),actual.getpixel((1152,312)))

    def test_request_has_hard_mask_and_opaque_reference_without_feather(self):
        generated=seamless.png(Image.new('RGB',(1536,1024)))
        with patch.object(seamless,'request_image',return_value=(generated,'req','gpt-image-2.5-flare',{})) as request:
            seamless.generate(self.source,'test','gpt-image-2.5-flare',180,True,None,cover_art)
        args=request.call_args.args
        with Image.open(BytesIO(args[3])) as canvas:
            self.assertEqual((25,110,200),canvas.convert('RGB').getpixel((461,205)))
        with Image.open(BytesIO(args[4])) as mask:
            self.assertEqual(255,mask.getpixel((461,205))[3])
            self.assertEqual(0,mask.getpixel((460,205))[3])
        recipe=json.loads(self.source.with_suffix('.recipe.json').read_text())
        self.assertEqual(('original',0),(recipe['shadow'],recipe['feather_px']))

    def test_frontier_compatibility_error_does_not_silently_downgrade(self):
        class Response:
            status_code = 404
            headers = {}
            def json(self):
                return {'error': {'code': 'model_not_found', 'message': 'Frontier model unavailable'}}
        with patch.object(seamless, 'request_image', side_effect=seamless.RequestFailure(Response())) as request:
            with self.assertRaisesRegex(seamless.RequestFailure, 'Frontier model unavailable'):
                seamless.generate(self.source, 'test', 'gpt-image-2.5-sunburst', 120, True, None, cover_art, allow_fallback=False)
        self.assertEqual(1, request.call_count)
        self.assertFalse(self.source.with_suffix('.recipe.json').exists())

    def test_transport_failure_does_not_try_another_model(self):
        with patch.object(seamless,'request_image',side_effect=requests.Timeout) as request:
            with self.assertRaises(requests.Timeout):
                seamless.generate(self.source,'test','gpt-image-2.5-flare',180,True,None,cover_art)
            self.assertEqual(1,request.call_count)

    def test_refusal_stops_fallback_but_unsupported_model_can_advance(self):
        class Response:
            status_code=400
            headers={}
            def json(self):return {'error':{'code':'moderation_blocked','message':'Safety system declined'}}
        self.assertFalse(seamless.RequestFailure(Response()).can_fallback())
        response=Response();response.json=lambda:{'error':{'code':'model_not_found','message':'Unsupported model'}}
        self.assertTrue(seamless.RequestFailure(response).can_fallback())


if __name__=='__main__':unittest.main()
