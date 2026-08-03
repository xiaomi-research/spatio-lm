import random
from typing import Any, Dict

import numpy as np
import torchvision.transforms as tvf
from swift.llm.template.template.qwen import Qwen2_5VLTemplate, Qwen2VLTemplate
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import findall
from swift.utils import get_env_args
from torchvision.transforms.functional import InterpolationMode

from ..utils.action_mask import ActionMasker


class Qwen2_5VLA0Template(Qwen2_5VLTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_masker = ActionMasker(self.tokenizer, mask_token="?")

        # Build image transforms
        self.image_transforms = tvf.Compose(
            [
                # tvf.RandomResizedCrop(
                #     (self.input_size, self.input_size),
                #     scale=(0.8, 1.2),
                #     interpolation=InterpolationMode.BICUBIC,
                # ),
                tvf.CenterCrop(self.input_size),
                tvf.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        )

    def init_env_args(self):
        super().init_env_args()

        # Random Mask Action to '?'
        self.action_mask_proba = get_env_args("action_mask_proba", float, 0.9)
        self.action_mask_ratio = get_env_args("action_mask_ratio", float, 0.4)

        # Align with original VLA0 implementation
        self.input_size = get_env_args("input_size", int, 224)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(Qwen2VLTemplate, self)._encode(inputs)
        processor = self.processor
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)
        for media_type in ["images", "videos"]:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == "images":
                    mm_data = [self.image_transforms(img) for img in mm_data]
                    inputs.images = mm_data

                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=mm_data, return_tensors="pt", do_resize=False
                    )
                    media_grid_thw = media_inputs["image_grid_thw"]
                else:
                    kwargs = {}
                    if hasattr(processor, "video_processor"):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs["images"] = None
                    media_inputs = processor_func(
                        videos=mm_data, return_tensors="pt", do_resize=False, **kwargs
                    )
                    media_grid_thw = media_inputs["video_grid_thw"]
                    media_token = self.video_token_id
                    if self.version == "v2_5":
                        fps = inputs.mm_processor_kwargs["fps"]
                        media_inputs["second_per_grid_ts"] = [
                            processor.image_processor.temporal_patch_size / tmp
                            for tmp in fps
                        ]
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = media_grid_thw[i].prod() // merge_length
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, idx_list, _get_new_tokens
                )
                encoded.update(media_inputs)

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale

        encoded = self.action_masker(
            encoded,
            self.action_mask_proba,
            self.action_mask_ratio,
        )

        return encoded
