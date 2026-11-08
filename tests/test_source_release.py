import copy
import asyncio
import gc
import unittest
import uuid
import weakref
from unittest import mock

from support import clip, extension, model, torch
import execution
import nodes
from comfy_execution.graph import ExecutionList


class Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *args, **kwargs):
        pass


class Source:
    calls = 0
    refs = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT",)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"

    def load(self, seed):
        type(self).calls += 1
        a, b = model(seed), clip(seed)
        type(self).refs.extend(weakref.ref(value) for value in (a, a.model, b, b.cond_stage_model))
        return a, b, torch.ones(4)


class Consumer:
    calls = 0
    assert_sources_gone = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "seed": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, model, clip, vae, seed):
        type(self).calls += 1
        if self.assert_sources_gone:
            gc.collect()
            assert all(ref() is None for ref in Source.refs), "Sources retained during downstream execution"
        assert torch.isfinite(model.model.diffusion_model(torch.ones(1, 8))).all()
        assert torch.isfinite(clip.cond_stage_model.diffusion_model(torch.ones(1, 8))).all()
        assert vae.sum() == 4
        return ()


class Branch:
    calls = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("MODEL",), "gate": ("MODEL",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, source, gate):
        type(self).calls += 1
        assert Source.refs[0]() is source
        assert torch.isfinite(source.model.diffusion_model(torch.ones(1, 8))).all()
        return ()


class AsyncBranch(Branch):
    async def run(self, source, gate):
        await asyncio.sleep(.01)
        return super().run(source, gate)


@unittest.skipUnless(getattr(ExecutionList, "OUTPUT_RELEASE_API_VERSION", None) == 1,
                     "Requires the reviewed ComfyUI core integration")
class SourceRelease(unittest.TestCase):
    def setUp(self):
        Source.calls = Consumer.calls = Branch.calls = 0
        AsyncBranch.calls = 0
        Consumer.assert_sources_gone = False
        Source.refs = []
        self.mapping = mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {
            **extension.NODE_CLASS_MAPPINGS, "TestSourceRelease": Source,
            "TestConsumerRelease": Consumer, "TestBranchRelease": Branch,
            "TestAsyncBranchRelease": AsyncBranch,
        })
        self.mapping.start()
        self.addCleanup(self.mapping.stop)
        self.prompt = {
            "1": {"class_type": "TestSourceRelease", "inputs": {"seed": 1}},
            "2": {"class_type": "TestSourceRelease", "inputs": {"seed": 2}},
            "3": {"class_type": "ModelMergeSimpleRelease", "inputs": {
                "model1": ["1", 0], "model2": ["2", 0], "ratio": .5, "release_sources": True}},
            "4": {"class_type": "CLIPMergeSimpleRelease", "inputs": {
                "clip1": ["1", 1], "clip2": ["2", 1], "ratio": .5, "release_sources": True}},
            "5": {"class_type": "TestConsumerRelease", "inputs": {
                "model": ["3", 0], "clip": ["4", 0], "vae": ["1", 2], "seed": 1}},
        }

    def execute(self, executor, outputs=("5",)):
        executor.execute(copy.deepcopy(self.prompt), str(uuid.uuid4()), execute_outputs=list(outputs))
        self.assertTrue(executor.success, executor.status_messages)

    def test_sources_collected_vae_kept_three_generations_both_caches(self):
        for policy in (execution.CacheType.CLASSIC, execution.CacheType.RAM_PRESSURE):
            with self.subTest(policy=policy):
                Source.calls = Consumer.calls = 0
                Source.refs = []
                self.prompt["5"]["inputs"]["seed"] = 1
                executor = execution.PromptExecutor(Server(), cache_type=policy,
                                                    cache_args={"ram": 1., "ram_inactive": 1.})
                self.execute(executor)
                gc.collect()
                self.assertEqual(Source.calls, 2)
                self.assertTrue(all(ref() is None for ref in Source.refs))
                cache = executor.caches.outputs
                self.assertIsNone(cache.get_local("1"))
                self.assertIsNone(cache.get_local("1", 0))
                self.assertIsNone(cache.get_local("1", 1))
                self.assertIsNotNone(cache.get_local("1", 2))
                final = cache.get_local("3").outputs[0][0]
                for seed in (2, 3):
                    self.prompt["5"]["inputs"]["seed"] = seed
                    self.execute(executor)
                    self.assertEqual(Source.calls, 2)
                    self.assertIs(cache.get_local("3").outputs[0][0], final)
                    cached = next(data["nodes"] for event, data in executor.status_messages if event == "execution_cached")
                    self.assertTrue({"3", "4"}.issubset(cached))
                    self.assertNotIn("1", cached)
                self.assertEqual(Consumer.calls, 3)
                del final, executor, cache
                gc.collect()

    def test_other_branch_finishes_before_source_is_released(self):
        self.prompt["6"] = {"class_type": "TestBranchRelease", "inputs": {"source": ["1", 0], "gate": ["3", 0]}}
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        self.execute(executor, ("5", "6"))
        self.assertEqual(Branch.calls, 1)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in Source.refs))

    def test_changed_ratio_reloads_released_inputs(self):
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        self.execute(executor)
        self.prompt["3"]["inputs"]["ratio"] = .25
        self.execute(executor)
        self.assertEqual(Source.calls, 4)
        self.assertEqual(Consumer.calls, 2)

    def test_toggle_off_restores_stock_behavior(self):
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        self.execute(executor)
        for node_id in ("3", "4"):
            self.prompt[node_id]["inputs"]["release_sources"] = False
        self.execute(executor)
        self.assertEqual(Source.calls, 4)
        self.assertIsNotNone(executor.caches.outputs.get_local("1", 0))
        self.assertIsNotNone(executor.caches.outputs.get_local("1", 1))

    def test_enabling_release_clears_inactive_lazy_variants(self):
        executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.RAM_PRESSURE,
                                            cache_args={"ram": 1., "ram_inactive": 1.})
        for node_id in ("3", "4"):
            self.prompt[node_id]["inputs"]["release_sources"] = False
        self.execute(executor)
        for node_id in ("3", "4"):
            self.prompt[node_id]["inputs"]["release_sources"] = True
        Consumer.assert_sources_gone = True
        self.execute(executor)
        self.assertEqual(Source.calls, 2)

    def test_async_branch_retains_source_until_completion(self):
        self.prompt["6"] = {"class_type": "TestAsyncBranchRelease", "inputs": {"source": ["1", 0], "gate": ["3", 0]}}
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        self.execute(executor, ("5", "6"))
        self.assertEqual(AsyncBranch.calls, 1)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in Source.refs))

    def test_shared_checkpoint_cache_key_preserves_branch(self):
        self.prompt["2"]["inputs"]["seed"] = 1
        self.prompt["6"] = {"class_type": "TestBranchRelease", "inputs": {"source": ["2", 0], "gate": ["3", 0]}}
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        self.execute(executor, ("5", "6"))
        self.assertEqual(Branch.calls, 1)
        self.assertEqual(Source.calls, 1)

    def test_downstream_exception_does_not_require_source_reload(self):
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        with self.assertLogs(level="ERROR"), mock.patch.object(Consumer, "run", side_effect=RuntimeError("test failure")):
            executor.execute(copy.deepcopy(self.prompt), str(uuid.uuid4()), execute_outputs=["5"])
            self.assertFalse(executor.success)
        self.prompt["5"]["inputs"]["seed"] = 2
        self.execute(executor)
        self.assertEqual(Source.calls, 2)

    def test_cache_entry_preserves_two_field_tuple_interface(self):
        from comfy_execution.caching import CacheEntry
        entry = CacheEntry(ui=None, outputs=[[1], [2]], producer=("1", "TestSourceRelease"))
        ui, outputs = entry
        self.assertIsNone(ui)
        self.assertEqual(outputs, [[1], [2]])
        trimmed = entry.without_outputs({0})
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed.released, frozenset({0}))
        self.assertEqual(trimmed._replace(ui={}).released, frozenset({0}))

    def test_release_trims_prompt_references_after_cache_eviction(self):
        original = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]

        class EvictAfterMaterialize(original):
            def merge(self, *args, **kwargs):
                result = super().merge(*args, **kwargs)
                cache = kwargs["execution_list"].output_cache
                for node_id in ("1", "2"):
                    cache.cache.pop(cache.cache_key_set.get_data_key(node_id), None)
                return result

        Consumer.assert_sources_gone = True
        executor = execution.PromptExecutor(Server(), cache_args={"ram": 1., "ram_inactive": 1.})
        with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"ModelMergeSimpleRelease": EvictAfterMaterialize}):
            self.execute(executor)
        self.assertEqual(Source.calls, 2)


class MissingIntegration(unittest.TestCase):
    def test_unpatched_executor_fails_before_materialization(self):
        from simplemerge_release.materialize import MaterializationError
        node = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]()
        with self.assertRaisesRegex(MaterializationError, "cache integration"):
            node.merge(model(1), model(2), .5, True, execution_list=object(), unique_id="3")
