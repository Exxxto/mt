"""
ADB controller module for the Merge Tactics bot.
Handles all Android Debug Bridge interactions.
"""

import os
import subprocess
import tempfile
import uuid
import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ADBActionResult:
    status: str
    return_code: int | None
    duration_ms: int


class ADBController:
    def __init__(self, adb_path, serial=None):
        self.adb_path = adb_path
        self.screenshot_mode = os.environ.get("ADB_SCREENSHOT_MODE", "exec-out").strip().lower()
        if self.screenshot_mode not in {"exec-out", "pull"}:
            raise ValueError("ADB_SCREENSHOT_MODE must be 'exec-out' or 'pull'")
        # BlueStacks exposes the same instance twice (emulator-5554 + 127.0.0.1:5555),
        # so a bare "adb" call fails with "more than one device/emulator".
        self.serial = serial or os.environ.get("ADB_SERIAL") or self._detect_serial()

    def _detect_serial(self):
        """Pick the device to control when more than one is attached"""
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
        except Exception as e:
            logger.warning(f"Could not list adb devices: {e}")
            return None

        devices = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])

        if not devices:
            logger.warning("No online adb device found")
            return None

        if len(devices) == 1:
            logger.info(f"Using adb device {devices[0]}")
            return devices[0]

        emulator = next((d for d in devices if d.startswith("emulator-")), devices[0])
        logger.warning(
            f"Multiple adb devices ({', '.join(devices)}), using {emulator}. "
            f"Set ADB_SERIAL to override."
        )
        return emulator

    def _adb_cmd(self, *args):
        """Build an adb command bound to the selected device"""
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd + list(args)

    def screenshot(self, timeout=None):
        """Take a screenshot using ADB"""
        if self.screenshot_mode == "pull":
            return self._screenshot_pull(timeout=timeout or 20)

        try:
            result = subprocess.run(
                self._adb_cmd("exec-out", "screencap", "-p"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise

        if not result.stdout:
            error = result.stderr.decode(errors="replace").strip() or "empty output"
            logger.error(f"Screenshot failed: {error}")
            raise RuntimeError(f"Screenshot failed: {error}")

        img = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Screenshot failed: could not decode image")
            raise RuntimeError("Screenshot failed: could not decode image")
        return img

    def _screenshot_pull(self, timeout=20):
        """Capture to a unique device file, pull it locally, and decode the PNG."""
        remote_path = f"/data/local/tmp/merge-tactics-{uuid.uuid4().hex}.png"
        try:
            with tempfile.TemporaryDirectory(prefix="merge-tactics-") as temp_dir:
                local_path = os.path.join(temp_dir, "screenshot.png")
                self._run_adb_checked(
                    self._adb_cmd("shell", "screencap", "-p", remote_path), "screencap", timeout=timeout
                )
                self._run_adb_checked(
                    self._adb_cmd("pull", remote_path, local_path), "pull", timeout=timeout
                )
                try:
                    with open(local_path, "rb") as screenshot_file:
                        png = screenshot_file.read()
                except FileNotFoundError as error:
                    raise RuntimeError("ADB pull did not create a screenshot file") from error

                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise RuntimeError("Screenshot failed: pulled file is not a PNG")
                image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("Screenshot failed: could not decode pulled PNG")
                return image
        finally:
            try:
                result = subprocess.run(
                    self._adb_cmd("shell", "rm", "-f", remote_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                )
                if result.returncode != 0:
                    logger.warning("Could not remove temporary ADB screenshot")
            except Exception as error:
                logger.warning("Could not remove temporary ADB screenshot: %s", type(error).__name__)

    @staticmethod
    def _run_adb_checked(command, operation, timeout):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            error = result.stderr.decode(errors="replace").strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"ADB {operation} failed: {error}")
        return result

    def tap(self, x, y, *, timeout=None, return_result=False):
        """Perform a tap at coordinates (x, y)"""
        started = time.monotonic()
        try:
            result = subprocess.run(
                self._adb_cmd("shell", "input", "tap", str(x), str(y)),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
            message = (result.stdout + result.stderr).decode(errors="replace").lower()
            status = "succeeded" if result.returncode == 0 else (
                "device_lost" if any(word in message for word in ("device not found", "device offline", "no devices"))
                else "nonzero_exit"
            )
            outcome = ADBActionResult(status, result.returncode,
                                      max(0, round((time.monotonic() - started) * 1000)))
            return outcome if return_result else None
        except subprocess.TimeoutExpired:
            if return_result:
                return ADBActionResult("timeout", None,
                                       max(0, round((time.monotonic() - started) * 1000)))
            raise
        except Exception as e:
            logger.error(f"Tap failed at ({x}, {y}): {e}")
            if return_result:
                return ADBActionResult("exception", None,
                                       max(0, round((time.monotonic() - started) * 1000)))
            raise
