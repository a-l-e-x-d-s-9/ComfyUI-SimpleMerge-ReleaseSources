from comfy_extras.nodes_model_merging import CLIPMergeSimple, ModelMergeSimple

from .diagnostics import log_materialization, snapshot
from .materialize import materialize_clip, materialize_model, validate_sources
from .source_release import prepare_release, schedule_release


class ModelMergeSimpleRelease:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = ModelMergeSimple.INPUT_TYPES()
        inputs["required"]["release_sources"] = ("BOOLEAN", {"default": False})
        inputs["hidden"] = {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"}
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/memory"

    def merge(self, model1, model2, ratio, release_sources=False, execution_list=None, unique_id=None):
        release = prepare_release(execution_list, unique_id) if release_sources else None
        before = snapshot() if release_sources else None
        merged = ModelMergeSimple().merge(model1, model2, ratio)[0]
        if release_sources:
            validate_sources(model1, model2)
            merged = materialize_model(merged)
            schedule_release(execution_list, release, "MODEL")
            log_materialization("MODEL", before)
        return (merged,)


class CLIPMergeSimpleRelease:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = CLIPMergeSimple.INPUT_TYPES()
        inputs["required"]["release_sources"] = ("BOOLEAN", {"default": False})
        inputs["hidden"] = {"execution_list": "EXECUTION_LIST", "unique_id": "UNIQUE_ID"}
        return inputs

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/memory"

    def merge(self, clip1, clip2, ratio, release_sources=False, execution_list=None, unique_id=None):
        release = prepare_release(execution_list, unique_id) if release_sources else None
        before = snapshot() if release_sources else None
        merged = CLIPMergeSimple().merge(clip1, clip2, ratio)[0]
        if release_sources:
            validate_sources(clip1.patcher, clip2.patcher)
            merged = materialize_clip(merged)
            schedule_release(execution_list, release, "CLIP")
            log_materialization("CLIP", before)
        return (merged,)
