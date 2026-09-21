# Modified for this inference-only distribution: restrict the vision factory to SigLIP2.
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    if vision_tower == "google/siglip2-so400m-patch14-384":
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    raise ValueError(f"Unsupported vision tower: {vision_tower}")
