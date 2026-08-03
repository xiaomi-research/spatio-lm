from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_intern_vit import InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internvl_chat import InternVLChatModel
from .modeling_internvl3r_chat import InternVL3RChatModel

AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)
AutoModel.register(InternVLChatConfig, InternVLChatModel)
AutoModel.register(InternVLChatConfig, InternVL3RChatModel)
AutoModelForCausalLM.register(InternVLChatConfig, InternVLChatModel)
AutoModelForCausalLM.register(InternVLChatConfig, InternVL3RChatModel)


__all__ = [
    "InternVisionConfig",
    "InternVLChatConfig",
    "InternVisionModel",
    "InternVLChatModel",
    "InternVL3RChatModel"
]
