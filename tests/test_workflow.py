import importlib.util
import unittest

from support import ROOT

spec = importlib.util.spec_from_file_location("converter", ROOT / "scripts/convert_workflow.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class Workflow(unittest.TestCase):
    def test_idempotence(self):
        original = {"nodes": [{"id": 2023, "type": "ModelMergeSimple", "widgets_values": [.75],
                                "widgets_values_named": {"ratio": .75}, "inputs": [], "outputs": [], "pos": [1, 2]},
                               {"id": 2022, "type": "CLIPMergeSimple", "widgets_values": [.75],
                                "inputs": [], "outputs": [], "pos": [3, 4]}], "links": []}
        out = converter.convert(original)
        self.assertEqual(out, converter.convert(out))
        self.assertEqual(original["nodes"][0]["widgets_values"], [.75])
        self.assertEqual(out["nodes"][0]["widgets_values"], [.75, True])
