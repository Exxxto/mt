import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.adb_controller import ADBController


class ADBScreenshotModeTests(unittest.TestCase):
    serial = "127.0.0.1:16416"

    @classmethod
    def setUpClass(cls):
        ok, encoded = cv2.imencode(".png", np.zeros((3, 4, 3), dtype=np.uint8))
        if not ok:
            raise RuntimeError("could not encode fixture PNG")
        cls.png = encoded.tobytes()

    def controller(self, mode=None):
        env = {"ADB_SERIAL": self.serial}
        if mode is not None:
            env["ADB_SCREENSHOT_MODE"] = mode
        else:
            env.pop("ADB_SCREENSHOT_MODE", None)
        return patch.dict(os.environ, env, clear=True)

    @staticmethod
    def completed(args, stdout=b"", returncode=0, stderr=b""):
        import subprocess
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    def test_default_mode_keeps_exec_out(self):
        with self.controller(), patch("core.adb_controller.subprocess.run") as run:
            run.return_value = self.completed([], stdout=self.png)
            image = ADBController("adb", serial=self.serial).screenshot()

        self.assertEqual(image.shape, (3, 4, 3))
        self.assertEqual(run.call_args.args[0], ["adb", "-s", self.serial, "exec-out", "screencap", "-p"])

    def test_invalid_mode_is_rejected_before_adb_access(self):
        with patch.dict(os.environ, {"ADB_SCREENSHOT_MODE": "unknown"}, clear=True):
            with patch("core.adb_controller.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "ADB_SCREENSHOT_MODE"):
                    ADBController("adb")

        run.assert_not_called()

    def test_pull_mode_downloads_valid_png_and_cleans_temporary_files(self):
        calls = []
        local_paths = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if "pull" in args:
                local = Path(args[-1])
                local_paths.append(local)
                local.write_bytes(self.png)
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            image = ADBController("adb").screenshot()

        self.assertEqual(image.shape, (3, 4, 3))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0][:6], ["adb", "-s", self.serial, "shell", "screencap", "-p"])
        self.assertEqual(calls[0][0][-1], calls[1][0][-2])
        self.assertEqual(calls[2][0][:6], ["adb", "-s", self.serial, "shell", "rm", "-f"])
        self.assertTrue(all(args[:3] == ["adb", "-s", self.serial] for args, _ in calls))
        self.assertTrue(all(call[1].get("timeout") for call in calls))
        self.assertTrue(all(not path.exists() for path in local_paths))

    def test_pull_uses_a_new_remote_path_for_each_frame(self):
        remote_paths = []

        def run(args, **kwargs):
            if "screencap" in args:
                remote_paths.append(args[-1])
            elif "pull" in args:
                Path(args[-1]).write_bytes(self.png)
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            adb = ADBController("adb", serial=self.serial)
            adb.screenshot()
            adb.screenshot()

        self.assertEqual(len(remote_paths), 2)
        self.assertNotEqual(*remote_paths)

    def test_pull_screencap_error_raises_and_attempts_remote_cleanup(self):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if "screencap" in args:
                return self.completed(args, returncode=1, stderr=b"capture failed")
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "screencap"):
                ADBController("adb", serial=self.serial).screenshot()

        self.assertIn("rm", calls[-1])

    def test_pull_error_raises_and_attempts_remote_cleanup(self):
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if "pull" in args:
                return self.completed(args, returncode=1, stderr=b"pull failed")
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "pull"):
                ADBController("adb", serial=self.serial).screenshot()

        self.assertIn("rm", calls[-1])

    def test_pull_missing_or_invalid_png_raises(self):
        for payload in (None, b"not a png"):
            with self.subTest(payload=payload), self.controller("pull"):
                def run(args, **kwargs):
                    if "pull" in args and payload is not None:
                        Path(args[-1]).write_bytes(payload)
                    return self.completed(args)

                with patch("core.adb_controller.subprocess.run", side_effect=run):
                    with self.assertRaises(RuntimeError):
                        ADBController("adb", serial=self.serial).screenshot()

    def test_pull_timeout_raises_and_attempts_remote_cleanup(self):
        import subprocess
        calls = []

        def run(args, **kwargs):
            calls.append(args)
            if "pull" in args:
                raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            with self.assertRaises(subprocess.TimeoutExpired):
                ADBController("adb", serial=self.serial).screenshot()

        self.assertIn("rm", calls[-1])

    def test_failed_second_frame_does_not_return_previous_frame(self):
        call_count = 0

        def run(args, **kwargs):
            nonlocal call_count
            if "screencap" in args:
                call_count += 1
                if call_count == 2:
                    return self.completed(args, returncode=1, stderr=b"capture failed")
            elif "pull" in args:
                Path(args[-1]).write_bytes(self.png)
            return self.completed(args)

        with self.controller("pull"), patch("core.adb_controller.subprocess.run", side_effect=run):
            adb = ADBController("adb", serial=self.serial)
            self.assertEqual(adb.screenshot().shape, (3, 4, 3))
            with self.assertRaises(RuntimeError):
                adb.screenshot()


if __name__ == "__main__":
    unittest.main()
