from copy import deepcopy
from typing import Dict, Optional, Tuple

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class VisionDptConfig(PretrainedConfig):
    model_type = "vision_dpt"

    def __init__(
        self,
        layer_ids: Tuple[int] = (-4, -3, -2, -1),
        dpt_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_ids = layer_ids

        if isinstance(dpt_config, dict):
            architecture: str = dpt_config["architectures"][0]
            if architecture == "DA3Model":
                from ..da3 import DA3Config
                self.dpt_config = DA3Config(**dpt_config)

            else:
                raise ValueError(f"Unsupport architecture: {architecture}")
        else:
            self.dpt_config = dpt_config

    def to_dict(self):
        output = deepcopy(self.__dict__)
        if self.dpt_config is not None:
            output["dpt_config"] = self.dpt_config.to_dict()

        return output
