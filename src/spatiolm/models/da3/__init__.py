from transformers import AutoConfig, AutoModel, AutoModelForDepthEstimation

from .configuration_da3 import DA3Config
from .modeling_da3 import DA3CameraHead, DA3DepthHead, DA3DinoBackbone, DA3Model

AutoConfig.register("depth_anything_3", DA3Config)
AutoModel.register(DA3Config, DA3Model)
AutoModelForDepthEstimation.register(DA3Config, DA3Model)


__all__ = [
    "DA3Config",
    "DA3Model",
    "DA3DinoBackbone",
    "DA3CameraHead",
    "DA3DepthHead",
]
