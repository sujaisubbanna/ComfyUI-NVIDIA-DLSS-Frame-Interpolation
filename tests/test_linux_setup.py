"""Exercise setup using real files, command peers and the generated launcher."""

import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "setup_linux", Path(__file__).parents[1] / "scripts/setup_linux.py"
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


@unittest.skipUnless(sys.platform == "linux", "Linux setup integration")
class LinuxSetupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dlss setup ' ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "node"
        self.prefix = self.root / "prefix"
        self.comfy = self.root / "ComfyUI"
        self.comfy.mkdir()
        (self.comfy / "main.py").write_text(
            "import os, sys, json\nfrom pathlib import Path\n"
            "Path('launched.json').write_text(json.dumps({"
            "'args':sys.argv[1:], 'prefix':os.environ['WINEPREFIX'],"
            "'overrides':os.environ['WINEDLLOVERRIDES']}))\n"
        )
        self.files = self.root / "proton/files"
        self.bin = self.files / "bin"
        self.bin.mkdir(parents=True)
        wine = (
            "import os\nfrom pathlib import Path\n"
            "p=Path(os.environ['WINEPREFIX'])\n"
            "(p/'drive_c/windows/system32').mkdir(parents=True, exist_ok=True)\n"
            "(p/'system.reg').touch()\n"
            "with (p/'boot.log').open('a') as f: f.write('called\\n')\n"
        )
        for name, body in {
            "wine": wine,
            "wineserver": "pass\n",
            "nvidia-smi": "print('NVIDIA GeForce RTX 5090')\n",
            "ffmpeg": "pass\n",
            "ffprobe": "pass\n",
        }.items():
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + body)
            path.chmod(0o755)
        self.compiler = self.root / "compiler.dll"
        self.driver = self.root / "driver"
        paths = [self.compiler, self.driver / "_nvngx.dll"]
        paths += [self.files / "lib" / p for p in setup.GRAPHICS.values()]
        paths += [
            self.repo / "bin/runtime" / p
            for p in (
                "host/nvngx.dll",
                "host/nvngx_dlssnr.dll",
                "host/renodx-dlss5.addon64",
                "host-linux/dxgi.dll",
                "dlss/nvngx_dlss.dll",
                "dlssg/dlssg-worker.exe",
                "dlssg/nvngx_dlssg.dll",
            )
        ]
        pe = b"MZ" + bytes(58) + struct.pack("<I", 64) + b"PE\0\0\x64\x86"
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pe)
        self.args = [
            "--proton",
            str(self.files.parent),
            "--comfyui",
            str(self.comfy),
            "--python",
            sys.executable,
            "--prefix",
            str(self.prefix),
            "--compiler",
            str(self.compiler),
            "--driver-dir",
            str(self.driver),
        ]
        self.enterContext(patch.object(setup, "ROOT", self.repo))
        self.enterContext(patch.dict(os.environ, {"PATH": str(self.bin)}))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_setup_rerun_and_launcher_with_quoted_paths(self):
        setup.main(self.args)
        system32 = self.prefix / "drive_c/windows/system32"
        victim = self.root / "outside.dll"
        victim.write_bytes(b"unchanged")
        (system32 / "dxgi.dll").unlink()
        (system32 / "dxgi.dll").symlink_to(victim)
        setup.main(self.args)
        self.assertEqual(victim.read_bytes(), b"unchanged")
        self.assertFalse((system32 / "dxgi.dll").is_symlink())
        self.assertEqual(
            (self.prefix / "boot.log").read_text(), "called\ncalled\n"
        )
        subprocess.run(
            [
                str(self.prefix / "start-comfyui.sh"),
                "--test-value",
                "$(touch unwanted); quote '",
            ],
            check=True,
            timeout=10,
        )
        result = json.loads((self.comfy / "launched.json").read_text())
        self.assertEqual(result["prefix"], str(self.prefix))
        self.assertEqual(
            result["args"], ["--test-value", "$(touch unwanted); quote '"]
        )
        self.assertIn("d3dcompiler_47=n,b", result["overrides"])
        self.assertFalse((self.comfy / "unwanted").exists())

    def test_check_makes_no_prefix_or_driver_links(self):
        setup.main([*self.args, "--check"])
        self.assertFalse(self.prefix.exists())
        self.assertFalse((self.repo / "bin/runtime/host/_nvngx.dll").exists())

    def test_existing_unmanaged_prefix_is_untouched(self):
        self.prefix.mkdir()
        sentinel = self.prefix / "system.reg"
        sentinel.write_text("game prefix")
        with self.assertRaisesRegex(RuntimeError, "unmanaged"):
            setup.main(self.args)
        self.assertEqual(sentinel.read_text(), "game prefix")
        self.assertFalse((self.prefix / setup.MARKER).exists())

    def test_lfs_pointer_fails_before_mutation(self):
        self.compiler.write_bytes(
            b"version https://git-lfs.github.com/spec/v1"
        )
        with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
            setup.main(self.args)
        self.assertFalse(self.prefix.exists())

    def test_wrong_architecture_fails_before_mutation(self):
        self.compiler.write_bytes(b"MZ" + bytes(62))
        with self.assertRaisesRegex(RuntimeError, "x86-64"):
            setup.main(self.args)
        self.assertFalse(self.prefix.exists())
