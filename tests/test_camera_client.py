import unittest
from unittest.mock import patch

from camera_client import Clock, normalize_server


class ServerAddressTests(unittest.TestCase):
    def test_bare_ip_gets_scheme_and_default_port(self):
        self.assertEqual(normalize_server("192.168.16.162"), "http://192.168.16.162:8765")
        self.assertEqual(normalize_server(" 192.168.16.162:9000/ "), "http://192.168.16.162:9000")
        self.assertEqual(normalize_server("http://192.168.16.162:8765"), "http://192.168.16.162:8765")
        self.assertEqual(normalize_server("http://servidor"), "http://servidor:8765")
        self.assertEqual(normalize_server("https://observa.lab"), "https://observa.lab")

    def test_invalid_address_is_rejected(self):
        for value in ("", "ftp://192.168.16.162", "http://"):
            with self.assertRaises(ValueError):
                normalize_server(value)


class Response:
    def __init__(self, remote): self.remote = remote
    def raise_for_status(self): pass
    def json(self): return {"monotonic":self.remote}


class FakeHTTP:
    def __init__(self): self.responses = iter([Response(105.1),Response(106.01),Response(107.2)])
    def get(self, *args, **kwargs): return next(self.responses)


class ClockTests(unittest.TestCase):
    def test_midpoint_offset_uses_lowest_rtt_and_renews(self):
        clock = Clock()
        with patch("camera_client.time.perf_counter", side_effect=[5,5.2,6,6.02,7,7.4]):
            clock.sync(FakeHTTP(),"http://server")
        timestamp, uncertainty = clock.timestamp(8)
        self.assertAlmostEqual(timestamp,108)
        self.assertAlmostEqual(uncertainty,0.01)

    def test_unsynchronized_clock_cannot_stamp_frames(self):
        with self.assertRaises(ValueError): Clock().timestamp(0)
