from swift.llm import TemplateType, register_model
from swift.llm.model.constant import MLLMModelType
from swift.llm.model.model.internlm import get_model_tokenizer_internvl
from swift.llm.model.model_arch import ModelArch
from swift.llm.model.register import Model, ModelGroup, ModelMeta
from swift.llm.template import register_template
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template.utils import ChatmlTemplateMeta
from swift.plugin.callback import extra_callbacks
from swift.utils import get_env_args
from swift.trainers.trainer_factory import TrainerFactory

from spatiolm.swift.callbacks import ModelParamsInfoCallback
from spatiolm.templates import InternvlV3RTemplate, InternvlVLA0Template, Qwen2_5VLA0Template


###############################################################################
#                           Register Trainer                                  #
###############################################################################
# TrainerFactory.TRAINER_MAPPING.update({
#     "causal_lm": "spatiolm.swift.trainers.Seq2Seq3DTrainer",
# })
# TrainerFactory.TRAINING_ARGS_MAPPING.update({
#     "causal_lm": "swift.trainers.Training3DArguments",
# })

###############################################################################
#                           Register Callback                                 #
###############################################################################
extra_callbacks.append(
    ModelParamsInfoCallback(
        info_depth=get_env_args("model_info_depth", int, 2)
    )
)

###############################################################################
#                            Register Templates                               #
###############################################################################
MLLMTemplateType.internvl3_5_v3r = "internvl3_5_v3r"
register_template(
    ChatmlTemplateMeta(MLLMTemplateType.internvl3_5_v3r, template_cls=InternvlV3RTemplate),
    exist_ok=True,
)

MLLMTemplateType.internvl3_5_vla0 = "internvl3_5_vla0"
register_template(
    ChatmlTemplateMeta(MLLMTemplateType.internvl3_5_vla0, template_cls=InternvlVLA0Template),
    exist_ok=True,
)

MLLMTemplateType.qwen2_5_vla0 = "qwen2_5_vla0"
register_template(
    ChatmlTemplateMeta(MLLMTemplateType.qwen2_5_vla0, template_cls=Qwen2_5VLA0Template),
    exist_ok=True,
)

###############################################################################
#                             Register Models                                 #
###############################################################################
MLLMModelType.internvl3_5_v3r = "internvl3_5_v3r"
register_model(
    ModelMeta(
        # TODO: Maybe add new type, like internvl3_5-v3r
        MLLMModelType.internvl3_5_v3r,
        # MLLMModelType.internvl3_5,
        [
            ModelGroup([
                Model("OpenGVLab/InternVL3_5-1B-V3R", "OpenGVLab/InternVL3_5-1B-V3R"),
                Model("OpenGVLab/InternVL3_5-2B-V3R", "OpenGVLab/InternVL3_5-2B-V3R"),
                Model("OpenGVLab/InternVL3_5-4B-V3R", "OpenGVLab/InternVL3_5-4B-V3R"),
                Model("OpenGVLab/InternVL3_5-8B-V3R", "OpenGVLab/InternVL3_5-8B-V3R"),
                Model("OpenGVLab/InternVL3_5-14B-V3R", "OpenGVLab/InternVL3_5-14B-V3R"),
                Model("OpenGVLab/InternVL3_5-38B-V3R", "OpenGVLab/InternVL3_5-38B-V3R"),
                Model("OpenGVLab/InternVL3_5-30B-A3B-V3R", "OpenGVLab/InternVL3_5-30B-A3B-V3R"),
                Model("OpenGVLab/InternVL3_5-241B-A28B-V3R", "OpenGVLab/InternVL3_5-241B-A28B-V3R"),
            ]),
        ],
        TemplateType.internvl3_5_v3r,
        get_model_tokenizer_internvl,
        architectures=["InternVL3RChatModel"],
        model_arch=ModelArch.internvl,
        requires=["transformers>=4.37.2"],
        tags=["vision", "video", "3R"],
    ),
    exist_ok=True,
)
