import gc
from unittest import mock
import unittest
import weakref

from support import clip, effective, extension, model, torch
from simplemerge_release.materialize import MaterializationError, materialize_model


class Independence(unittest.TestCase):
    def test_storage_and_collection(self):
        a, b = model(1), model(2)
        source_refs = [weakref.ref(x) for x in (a, b, a.model, b.model)]
        sources = {t.untyped_storage().data_ptr() for p in (a, b) for t in p.model.state_dict().values()}
        out = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]().merge(a, b, .5, True)[0]
        for t in out.model.state_dict().values():
            self.assertNotIn(t.untyped_storage().data_ptr(), sources)
        self.assertIsNone(out.parent)
        self.assertIsNone(out.cached_patcher_init)
        self.assertFalse(out.backup)
        self.assertFalse(out.object_patches_backup)
        del a, b
        gc.collect()
        self.assertTrue(all(ref() is None for ref in source_refs))
        self.assertTrue(torch.isfinite(out.model.diffusion_model(torch.ones(1, 8))).all())

    def test_clip_tokenizer_is_owned(self):
        a, b = clip(1), clip(2)
        refs = [weakref.ref(x) for x in (a, b, a.cond_stage_model, b.cond_stage_model)]
        out = extension.NODE_CLASS_MAPPINGS["CLIPMergeSimpleRelease"]().merge(a, b, .5, True)[0]
        self.assertIs(out.patcher.model, out.cond_stage_model)
        self.assertIsNot(out.tokenizer, a.tokenizer)
        self.assertNotEqual(out.tokenizer.data["embedding"].data_ptr(), a.tokenizer.data["embedding"].data_ptr())
        del a, b
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_loaded_source_backup_and_future_lora(self):
        a, b = model(1), model(2)
        a.add_patches({"diffusion_model.weight": (torch.ones(8, 8),)}, .2)
        before = effective(a, "diffusion_model.weight")
        a.patch_model(device_to=torch.device("cpu"))
        loaded = a.model.diffusion_model.weight.detach().clone()
        out = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]().merge(a, b, .5, True)[0]
        from comfy_extras.nodes_model_merging import ModelMergeSimple
        native = ModelMergeSimple().merge(a, b, .5)[0]
        torch.testing.assert_close(out.model.diffusion_model.weight, effective(native, "diffusion_model.weight"), rtol=0, atol=0)
        torch.testing.assert_close(a.model.diffusion_model.weight, loaded)
        original_out = out.model.diffusion_model.weight.detach().clone()
        patched = out.clone()
        # Native low-rank LoRA tuple, then ordinary patch/load/unpatch cycle.
        import comfy.lora
        lora = comfy.lora.load_lora({"test.lora_up.weight": torch.ones(8, 2),
                                     "test.lora_down.weight": torch.ones(2, 8)},
                                    {"test": "diffusion_model.weight"})
        self.assertIn("diffusion_model.weight", lora)
        patched.add_patches(lora, .1)
        patched.patch_model(device_to=torch.device("cpu"))
        self.assertFalse(torch.equal(patched.model.diffusion_model.weight, original_out))
        patched.unpatch_model()
        torch.testing.assert_close(out.model.diffusion_model.weight, original_out)
        a.unpatch_model()
        torch.testing.assert_close(effective(a, "diffusion_model.weight"), before)

    def test_cancellation_and_error_leave_inputs(self):
        a = model(1)
        before = {k: t.clone() for k, t in a.model.state_dict().items()}
        with mock.patch("comfy.model_management.throw_exception_if_processing_interrupted", side_effect=[None, None, RuntimeError("cancelled")]):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                materialize_model(a)
        a.model_options["wrapper"] = lambda: a
        with self.assertRaises(MaterializationError):
            materialize_model(a)
        for key, tensor in a.model.state_dict().items():
            torch.testing.assert_close(tensor, before[key])

    def test_unsupported_and_memory(self):
        a = model(1)
        a.attachments["test"] = object()
        with self.assertRaisesRegex(MaterializationError, "attachments"):
            materialize_model(a)
        a.attachments.clear()
        with mock.patch("simplemerge_release.materialize.psutil") as memory:
            memory.virtual_memory.return_value.available = 0
            with self.assertRaisesRegex(MaterializationError, "free RAM"):
                materialize_model(a)

    def test_lowvram_runtime_functions_are_removed(self):
        import comfy.ops
        a, b = model(1), model(2)
        layer = comfy.ops.manual_cast.Linear(8, 8)
        layer.weight = a.model.diffusion_model.weight
        layer.bias = a.model.diffusion_model.bias
        a.model.diffusion_model = layer
        a.add_patches({"diffusion_model.weight": (torch.ones(8, 8),)}, .2)
        a.patch_model(device_to=torch.device("cpu"), lowvram_model_memory=1)
        self.assertTrue(a.model.diffusion_model.weight_function)
        out = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]().merge(a, b, .5, True)[0]
        self.assertFalse(out.model.diffusion_model.weight_function)
        self.assertTrue(a.model.diffusion_model.weight_function)
        a.unpatch_model()

    def test_native_nan_values_are_preserved(self):
        a = model(1)
        with torch.no_grad():
            a.model.diffusion_model.weight[0, 0] = float("nan")
        with self.assertLogs(level="WARNING"):
            out = materialize_model(a)
        torch.testing.assert_close(a.model.diffusion_model.weight, out.model.diffusion_model.weight, equal_nan=True)

    def test_native_tokenizer_decoder_does_not_capture_source(self):
        import comfy.sdxl_clip
        from simplemerge_release.materialize import _copy_owned
        source = comfy.sdxl_clip.SDXLTokenizer()
        ref = weakref.ref(source)
        owned = _copy_owned(source)
        text = "a peaceful landscape (in watercolor:1.2)"
        self.assertEqual(source.tokenize_with_weights(text), owned.tokenize_with_weights(text))
        self.assertIsNot(source.clip_l.tokenizer.backend_tokenizer, owned.clip_l.tokenizer.backend_tokenizer)
        decoder = owned.clip_l.tokenizer.backend_tokenizer.decode
        backend_method = next(cell.cell_contents for cell in decoder.__closure__ if callable(cell.cell_contents))
        self.assertIs(backend_method.__self__, owned.clip_l.tokenizer.backend_tokenizer)
        del source
        gc.collect()
        self.assertIsNone(ref())

    def test_repeated_outputs_are_collectable(self):
        a, b = clip(1), clip(2)
        for _ in range(3):
            out = extension.NODE_CLASS_MAPPINGS["CLIPMergeSimpleRelease"]().merge(a, b, .5, True)[0]
            refs = [weakref.ref(x) for x in (out, out.patcher, out.cond_stage_model)]
            del out
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))

    def test_weight_replacing_object_patch_is_rejected(self):
        a = model(1)
        a.add_object_patch("diffusion_model", torch.nn.Linear(8, 8))
        with self.assertRaisesRegex(MaterializationError, "replaces registered weights"):
            materialize_model(a)
