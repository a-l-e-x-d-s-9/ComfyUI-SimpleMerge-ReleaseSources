import asyncio
import copy
from unittest import mock
import unittest

from support import extension, model
import nodes
from comfy_execution.caching import CacheKeySetInputSignature, HierarchicalCache, RAMPressureCache
from comfy_execution.graph import DynamicPrompt


class Changed:
    async def get(self, node_id):
        return False


class CacheReuse(unittest.TestCase):
    def test_seed_prompt_and_toggle_signatures(self):
        async def check(cache_class):
            dummy = type("Dummy", (), {"INPUT_TYPES": classmethod(lambda cls: {})})
            mapping = {**extension.NODE_CLASS_MAPPINGS, "Source": dummy, "Generation": dummy}
            with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping):
                prompt = {"1": {"class_type": "Source", "inputs": {"checkpoint": "a"}},
                          "2": {"class_type": "Source", "inputs": {"checkpoint": "b"}},
                          "3": {"class_type": "ModelMergeSimpleRelease", "inputs": {
                              "model1": ["1", 0], "model2": ["2", 0], "ratio": .5, "release_sources": True}},
                          "4": {"class_type": "Generation", "inputs": {"model": ["3", 0], "seed": 0, "text": "one"}}}
                cache = cache_class(CacheKeySetInputSignature)
                executions = 0
                sources = model(1), model(2)
                for seed in range(3):
                    prompt["4"]["inputs"].update(seed=seed, text=str(seed))
                    await cache.set_prompt(DynamicPrompt(copy.deepcopy(prompt)), prompt.keys(), Changed())
                    cache.clean_unused()
                    output = await cache.get("3")
                    if output is None:
                        executions += 1
                        output = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]().merge(*sources, .5, True)
                        await cache.set("3", output)
                    self.assertIsNone(await cache.get("4"))
                self.assertEqual(executions, 1)
                prompt["3"]["inputs"]["release_sources"] = False
                await cache.set_prompt(DynamicPrompt(prompt), prompt.keys(), Changed())
                self.assertIsNone(await cache.get("3"))
        for cls in (HierarchicalCache, RAMPressureCache):
            with self.subTest(cache=cls.__name__):
                asyncio.run(check(cls))
