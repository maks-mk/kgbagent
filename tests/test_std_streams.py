import sys
import unittest
from unittest import mock

from core.std_streams import ensure_standard_streams

_STREAM_NAMES = ("stdin", "stdout", "stderr")


class EnsureStandardStreamsTests(unittest.TestCase):
    def test_none_streams_are_replaced_with_working_null_devices(self):
        originals = {name: getattr(sys, name) for name in _STREAM_NAMES}
        created = {}
        try:
            for name in _STREAM_NAMES:
                setattr(sys, name, None)
            ensure_standard_streams()
            created = {name: getattr(sys, name) for name in _STREAM_NAMES}
        finally:
            for name, original in originals.items():
                setattr(sys, name, original)

        self.assertTrue(all(stream is not None for stream in created.values()))
        self.assertEqual(created["stdout"].write(""), 0)
        self.assertEqual(created["stderr"].write(""), 0)
        created["stdout"].flush()
        self.assertEqual(created["stdin"].read(0), "")
        for stream in created.values():
            stream.close()

    def test_existing_streams_are_left_untouched(self):
        for name in _STREAM_NAMES:
            with self.subTest(stream=name):
                sentinel = object()
                with mock.patch.object(sys, name, sentinel):
                    ensure_standard_streams()
                    self.assertIs(getattr(sys, name), sentinel)


if __name__ == "__main__":
    unittest.main()
