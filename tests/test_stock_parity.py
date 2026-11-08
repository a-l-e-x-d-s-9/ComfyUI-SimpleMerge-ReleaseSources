import unittest

from support import clip, effective, extension, model, torch
from comfy_extras.nodes_model_merging import CLIPMergeSimple, ModelMergeSimple


class StockParity(unittest.TestCase):
    def compare(self, reference, actual):
        if hasattr(reference, "patcher"):
            reference, actual = reference.patcher, actual.patcher
        self.assertEqual(set(reference.model.state_dict()), set(actual.model.state_dict()))
        for key, tensor in actual.model.state_dict().items():
            torch.testing.assert_close(effective(reference, key), tensor, rtol=0, atol=0)

    def test_all_ratios_and_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for factory, stock, new in ((model, ModelMergeSimple, extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]),
                                        (clip, CLIPMergeSimple, extension.NODE_CLASS_MAPPINGS["CLIPMergeSimpleRelease"])):
                a, b = factory(3, dtype), factory(7, dtype)
                for ratio in (0., .25, .5, .75, 1.):
                    with self.subTest(dtype=dtype, factory=factory.__name__, ratio=ratio):
                        native = stock().merge(a, b, ratio)[0]
                        self.compare(native, new().merge(a, b, ratio, True)[0])
                        lazy = new().merge(a, b, ratio, False)[0]
                        p = lazy.patcher if hasattr(lazy, "patcher") else lazy
                        source = a.patcher if hasattr(a, "patcher") else a
                        self.assertIs(p.model, source.model)
                        for key in p.model.state_dict():
                            native_p = native.patcher if hasattr(native, "patcher") else native
                            torch.testing.assert_close(effective(p, key), effective(native_p, key), rtol=0, atol=0)

    def test_chained_prior_patches_and_missing_keys(self):
        for factory, stock, new in ((model, ModelMergeSimple, extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]),
                                    (clip, CLIPMergeSimple, extension.NODE_CLASS_MAPPINGS["CLIPMergeSimpleRelease"])):
            a, b, c = factory(1), factory(2), factory(3)
            ap = a.patcher if hasattr(a, "patcher") else a
            bp = b.patcher if hasattr(b, "patcher") else b
            ap.add_patches({"diffusion_model.weight": (torch.ones(8, 8),)}, .12)
            bp.add_patches({"diffusion_model.bias": (torch.ones(8),)}, .31)
            native = stock().merge(stock().merge(a, b, .25)[0], c, .75)[0]
            actual = new().merge(new().merge(a, b, .25, False)[0], c, .75, True)[0]
            self.compare(native, actual)
            p = actual.patcher if hasattr(actual, "patcher") else actual
            self.assertFalse(p.patches)
            if hasattr(actual, "tokenizer"):
                self.assertEqual(actual.layer_idx, -2)
                torch.testing.assert_close(actual.cond_stage_model.special.logit_scale,
                                           a.cond_stage_model.special.logit_scale)

    def test_schema(self):
        for name, cls in extension.NODE_CLASS_MAPPINGS.items():
            keys = list(cls.INPUT_TYPES()["required"])
            self.assertEqual(keys[-2:], ["ratio", "release_sources"])
            self.assertFalse(hasattr(cls, "IS_CHANGED"))
            self.assertFalse(getattr(cls, "OUTPUT_NODE", False))

    def test_prior_native_lora_is_baked(self):
        import comfy.lora
        a, b, c = model(1), model(2), model(3)
        patches = comfy.lora.load_lora({"test.lora_up.weight": torch.ones(8, 2),
                                        "test.lora_down.weight": torch.ones(2, 8)},
                                       {"test": "diffusion_model.weight"})
        a.add_patches(patches, .3)
        b.add_patches(patches, .7)
        first = ModelMergeSimple().merge(a, b, .25)[0]
        native = ModelMergeSimple().merge(first, c, .75)[0]
        out = extension.NODE_CLASS_MAPPINGS["ModelMergeSimpleRelease"]().merge(first, c, .75, True)[0]
        self.compare(native, out)
