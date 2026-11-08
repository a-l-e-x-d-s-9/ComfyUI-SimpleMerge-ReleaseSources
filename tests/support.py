import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_PATH", ROOT.parents[1]))
sys.path.insert(0, str(COMFY))

import comfy.cli_args

comfy.cli_args.args.cpu = True
comfy.cli_args.args.disable_dynamic_vram = True

spec = importlib.util.spec_from_file_location("simplemerge_release", ROOT / "__init__.py",
                                             submodule_search_locations=[str(ROOT)])
extension = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extension
spec.loader.exec_module(extension)

import torch
import comfy.model_patcher
import comfy.sd


class TinyModel(torch.nn.Module):
    def __init__(self, seed=0, dtype=torch.float32):
        super().__init__()
        self.diffusion_model = torch.nn.Linear(8, 8, dtype=dtype)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=generator).to(dtype))
        self.register_buffer("outside_diffusion", torch.tensor([seed], dtype=dtype))
        self.register_buffer("temporary", torch.tensor([seed + 100.]), persistent=False)


class TinyTokenizer:
    def __init__(self):
        self.data = {"embedding": torch.ones(3)}


def model(seed=0, dtype=torch.float32):
    return comfy.model_patcher.ModelPatcher(TinyModel(seed, dtype), torch.device("cpu"), torch.device("cpu"))


def clip(seed=0, dtype=torch.float32):
    result = comfy.sd.CLIP(no_init=True)
    module = TinyModel(seed, dtype)
    module.special = torch.nn.Module()
    module.special.register_buffer("position_ids", torch.arange(3))
    module.special.register_buffer("logit_scale", torch.tensor(float(seed)))
    result.patcher = comfy.model_patcher.ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))
    result.patcher.is_clip = True
    result.cond_stage_model = module
    result.tokenizer = TinyTokenizer()
    result.layer_idx = -2
    result.tokenizer_options = {"example": [1]}
    result.use_clip_schedule = False
    result.apply_hooks_to_conds = None
    return result


def effective(patcher, key):
    # Independent reference: evaluate the installed native patch list, including
    # backup selection and native per-key dtype rounding.
    import comfy.float
    import comfy.lora
    import comfy.utils
    value = patcher.get_key_patches()[key]
    weight = value[0][0].detach().to(device="cpu", copy=True)
    if key in patcher.patches:
        weight = comfy.lora.calculate_weight(value[1:], weight.float(), key)
        weight = comfy.float.stochastic_rounding(weight, value[0][0].dtype,
                                                 seed=comfy.utils.string_to_seed(key))
    return weight
