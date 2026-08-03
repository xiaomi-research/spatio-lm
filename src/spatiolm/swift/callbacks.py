from contextlib import nullcontext

import pandas as pd
from swift.utils import get_logger
from transformers import TrainerCallback
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer import unwrap_model

from spatiolm.utils.torchinfo import summarize_model

__all__ = ["ModelParamsInfoCallback"]


logger = get_logger()


class ModelParamsInfoCallback(TrainerCallback):
    def __init__(self, info_depth=1):
        self.info_depth = info_depth

    def on_train_begin(self, args, state, control, **kwargs):
        model = unwrap_model(kwargs["model"])
        if is_deepspeed_zero3_enabled():
            import deepspeed

            context = deepspeed.zero.GatheredParameters(list(model.parameters()))
        else:
            context = nullcontext()
        with context:
            model_info = (
                summarize_model(model)
                .query(f"Depth <= {self.info_depth}")
                # .drop(columns=["Depth"])
                .set_index("Module")
            )
        with pd.option_context(
            "display.max_rows",
            200,
            "display.max_columns",
            20,
            "display.expand_frame_repr",
            False,
        ):
            logger.info(f"Model parameters: \n{model_info}")
