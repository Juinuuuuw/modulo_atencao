import unittest

from nao_client import make_request, encode_image, FrameReader
import io
from PIL import Image


class NaoRequestTests(unittest.TestCase):
    def test_rgb_body_is_preserved_as_bytes(self):
        payload = bytes([0,128,255]) * 20
        request = make_request("http://localhost:8765/api/nodes/nao/frame/0",payload,
                               {"Content-Type":"application/x-rgb", "X-Width":20})
        self.assertIs(request.data,payload)
        self.assertEqual(request.get_header("X-width"),"20")
        self.assertEqual(request.full_url,"http://localhost:8765/api/nodes/nao/frame/0")

    def test_jpeg_retains_dimensions_and_rgb_channel_order(self):
        frame=[80,60,3,11,0,0,bytes([230,30,10])*80*60]
        data,mime=encode_image(frame)
        self.assertEqual(mime,"image/jpeg")
        image=Image.open(io.BytesIO(data))
        self.assertEqual(image.size,(80,60))
        red,green,blue=image.getpixel((40,30))
        self.assertGreater(red,200)
        self.assertLess(blue,30)
        self.assertLess(len(data),len(frame[6]))

    def test_raw_fallback_preserves_binary(self):
        frame=[2,1,3,11,0,0,bytes([0,128,255])*2]
        data,mime=encode_image(frame,raw_rgb=True)
        self.assertEqual(data,frame[6])
        self.assertEqual(mime,"application/x-rgb")

    def test_capture_failure_is_reported_to_sender(self):
        class BrokenVideo:
            def getImageRemote(self, subscription):
                raise RuntimeError("Camera disconnected")
        reader=FrameReader(BrokenVideo(),None,"test")
        reader.start()
        reader.join(timeout=1)
        with self.assertRaisesRegex(RuntimeError,"Camera disconnected"):
            reader.get_latest()
        reader.close()
