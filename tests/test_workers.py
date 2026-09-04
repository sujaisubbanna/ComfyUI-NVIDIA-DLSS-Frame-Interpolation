import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dlss_engine.core.workers import (
    WorkerProcess,
    validate_binary,
    worker_launch,
)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dlss test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = self.root / "worker.exe"
        self.worker.write_bytes(b"MZ" + bytes(128))
        self.prefix = self.root / "prefix"
        (self.prefix / "drive_c/windows/system32").mkdir(parents=True)

    def test_windows_launch_is_direct(self):
        with patch("dlss_engine.core.workers.sys.platform", "win32"):
            command, options = worker_launch(self.worker, "--probe")
        self.assertEqual(command, [str(self.worker), "--probe"])
        self.assertNotIn("start_new_session", options)

    @unittest.skipUnless(sys.platform == "linux", "Linux configuration")
    def test_linux_requires_configured_prefix(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WINEPREFIX"):
                worker_launch(self.worker, "--probe")

    @unittest.skipUnless(sys.platform == "linux", "Linux configuration")
    def test_missing_wine_is_actionable(self):
        env = {"WINEPREFIX": str(self.prefix), "DLSS_WINE_PATH": "/missing"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DLSS_WINE_PATH"):
                worker_launch(self.worker, "--probe")

    def test_lfs_pointer_and_invalid_binary_fail_before_launch(self):
        self.worker.write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
            validate_binary(self.worker)
        self.worker.write_text("not a Windows executable")
        with self.assertRaisesRegex(RuntimeError, "Windows"):
            validate_binary(self.worker)

    @unittest.skipUnless(sys.platform == "linux", "Linux subprocess flow")
    def test_binary_streams_and_paths_with_spaces(self):
        # A real subprocess stands in for Wine; no shell parses these paths.
        launcher = self.root / "wine runner"
        launcher.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "assert sys.argv[1].endswith('worker.exe')\n"
            "assert sys.argv[2] == '--serve'\n"
            "assert os.environ['WINEPREFIX'].endswith('prefix')\n"
            "sys.stderr.write('worker diagnostic\\n')\n"
            "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
        )
        launcher.chmod(0o755)
        env = {
            "WINEPREFIX": str(self.prefix),
            "DLSS_WINE_PATH": str(launcher),
        }
        payload = bytes(range(256)) * 8192
        with patch.dict(os.environ, env):
            with WorkerProcess(
                self.worker, "--serve", stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ) as process:
                output, errors = process.communicate(payload, timeout=10)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(output, payload)
        self.assertEqual(errors, b"worker diagnostic\n")

    @unittest.skipUnless(sys.platform == "linux", "Linux process groups")
    def test_cancel_stops_child_without_stopping_unrelated_process(self):
        launcher = self.root / "wine"
        launcher.write_text(
            f"#!{sys.executable}\n"
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        launcher.chmod(0o755)
        env = {
            "WINEPREFIX": str(self.prefix),
            "DLSS_WINE_PATH": str(launcher),
        }
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        self.addCleanup(unrelated.wait)
        self.addCleanup(unrelated.kill)
        with patch.dict(os.environ, env):
            process = WorkerProcess(self.worker, stdout=subprocess.PIPE)
        try:
            self.assertGreater(int(process.stdout.readline()), 0)
            process.terminate()
            process.wait(timeout=5)
            # EOF proves the child no longer holds the inherited pipe open.
            output, _ = process.communicate(timeout=5)
            self.assertEqual(output, b"")
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertIsNone(unrelated.poll())
        finally:
            process.kill()
            process.wait(timeout=5)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
