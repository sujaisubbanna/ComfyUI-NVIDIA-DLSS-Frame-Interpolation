from __future__ import annotations

import ast
import subprocess
from pathlib import Path
import unittest

# Type helpers kept simple to keep this test dependency-light and runnable without Comfy.

def _extract_io_calls(expr, fn_defs):
    if isinstance(expr, (ast.List, ast.Tuple)):
        result = []
        for item in expr.elts:
            result.extend(_extract_io_calls(item, fn_defs))
        return result

    if not isinstance(expr, ast.Call):
        return []

    # Expand helper calls used in schema definitions
    if isinstance(expr.func, ast.Name) and expr.func.id in fn_defs:
        return _extract_io_calls(fn_defs[expr.func.id], fn_defs)

    if isinstance(expr.func, ast.Attribute) and expr.func.attr in {"Input", "Output"}:
        arg0 = expr.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            return [arg0.value]

    return []


def _collect_node_contracts(source: str) -> dict[str, dict[str, list[str]]]:
    tree = ast.parse(source)
    fn_defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list and node.name in {
            "_neural_inputs",
            "_output_detail_strength_input",
        }:
            if isinstance(node, ast.FunctionDef):
                returns = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        returns = stmt.value
                        break
                if returns is not None:
                    fn_defs[node.name] = returns

    contracts = {}
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        for item in cls.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "define_schema":
                continue
            for stmt in item.body:
                if not isinstance(stmt, ast.Return):
                    continue
                call = stmt.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "Schema"):
                    continue
                schema_kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
                contracts[cls.name] = {
                    "inputs": _extract_io_calls(schema_kwargs.get("inputs", ast.List()), fn_defs),
                    "outputs": _extract_io_calls(schema_kwargs.get("outputs", ast.List()), fn_defs),
                }
    return contracts


class ImageNodeInterfaceTests(unittest.TestCase):
    @staticmethod
    def _read_main_init(root: Path):
        candidates = [
            "upstream/main",
            "origin/main",
            "main",
        ]
        for ref in candidates:
            try:
                return subprocess.check_output(
                    ["git", "show", f"{ref}:__init__.py"],
                    cwd=root,
                    text=True,
                )
            except subprocess.CalledProcessError:
                continue
        return ""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parents[1]
        current = (root / "__init__.py").read_text()
        main = cls._read_main_init(root)
        cls.current_contracts = _collect_node_contracts(current)
        cls.main_contracts = _collect_node_contracts(main) if main else {}
        cls.current_source = current
        cls.has_main_reference = bool(main)

    def test_legacy_node_input_contracts_match_upstream(self):
        if not self.has_main_reference:
            self.skipTest("No upstream/main reference available to compare upstream node contract compatibility")
        for node_name in [
            "NvidiaDLSSFrameInterpolation",
            "NvidiaDLSSVideoUpscale",
            "NvidiaDLSSImageUpscale",
        ]:
            self.assertEqual(
                self.current_contracts[node_name]["inputs"],
                self.main_contracts[node_name]["inputs"],
                f"{node_name} input contract does not match upstream main",
            )

    def test_registered_node_ids_and_contracts(self):
        for node_name in [
            "NvidiaDLSSFrameInterpolation",
            "NvidiaDLSSVideoUpscale",
            "NvidiaDLSSImageUpscale",
            "NvidiaDLSSImageFrameInterpolation",
            "NvidiaDLSSImageSequenceUpscale",
        ]:
            self.assertIn(f'node_id="{node_name}"', self.current_source)

        self.assertEqual(
            self.current_contracts["NvidiaDLSSFrameInterpolation"]["outputs"],
            ["video", "report"],
        )
        self.assertEqual(
            self.current_contracts["NvidiaDLSSVideoUpscale"]["outputs"],
            ["video", "report"],
        )
        self.assertEqual(
            self.current_contracts["NvidiaDLSSImageUpscale"]["outputs"],
            ["image", "report"],
        )
        self.assertIn("images", self.current_contracts["NvidiaDLSSImageFrameInterpolation"]["outputs"])
        self.assertIn("output_fps", self.current_contracts["NvidiaDLSSImageFrameInterpolation"]["outputs"])

    def test_preview_output_uses_ui_preview_namespace(self):
        self.assertIn("ui.PreviewVideo([preview])", self.current_source)
        self.assertNotIn("io.PreviewVideo([preview])", self.current_source)

    def test_default_input_fps_is_supported(self):
        source = (Path(__file__).parents[1] / "dlss_engine" / "frame_interpolation" / "models.py").read_text()
        self.assertIn('"24": Fraction(24, 1)', source)


if __name__ == "__main__":
    unittest.main()
