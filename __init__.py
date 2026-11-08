from .nodes import CLIPMergeSimpleRelease, ModelMergeSimpleRelease

NODE_CLASS_MAPPINGS = {
    "ModelMergeSimpleRelease": ModelMergeSimpleRelease,
    "CLIPMergeSimpleRelease": CLIPMergeSimpleRelease,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeSimpleRelease": "Model Merge Simple (Release Sources)",
    "CLIPMergeSimpleRelease": "CLIP Merge Simple (Release Sources)",
}
