# Modified for this inference-only distribution: retain the checkpoint's MLP projector.
import torch.nn as nn


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", None)
    if projector_type != "mlp2x_gelu":
        raise ValueError(f"Unsupported projector type: {projector_type}")
    return nn.Sequential(
        nn.Linear(config.mm_hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
    )
