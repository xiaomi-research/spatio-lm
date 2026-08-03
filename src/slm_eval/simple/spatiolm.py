import importlib
import os
from typing import List

import torch
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl2 import (
    DEFAULT_GEN_KWARGS,
    InternVL2,
    load_image,
    load_video,
)
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 448))
NUM_FRAME = int(os.environ.get("VIDEO_SEGMENTS", 48))


@register_model("spatiolm")
class SpatioLM(InternVL2):
    def __init__(self, *args, **kwargs):
        # Register the InternVL3RChatModel to the AutoModel class in Transformers
        importlib.import_module("spatiolm.models")
        kwargs.setdefault("num_frame", NUM_FRAME)
        super().__init__(*args, **kwargs)

    def _get_modality(self, visuals: list):
        if all(isinstance(visual, Image.Image) for visual in visuals):
            return "image"
        elif all(
            isinstance(visual, str)
            and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".flv"))
            for visual in visuals
        ):
            return "video"
        else:
            eval_logger.warning(
                "Detected unsupported modality. Using default modality."
            )
            return self.modality

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            # Auto determine the modality of the input visuals
            modality = self._get_modality(visuals)

            if modality == "image":
                if visuals:
                    visuals = [
                        load_image(visual, IMAGE_SIZE).to(torch.bfloat16).cuda()
                        for visual in visuals
                    ]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals]
                    image_tokens = ["<image>"] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    contexts,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            elif modality == "video":
                assert len(visuals) == 1, (
                    f"Only one video is supported, but got {len(visuals)} videos."
                )
                video_path = visuals[0]
                pixel_values, num_patches_list = load_video(
                    video_path, input_size=IMAGE_SIZE, num_segments=self.num_frame
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = "".join(
                    [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
                )
                question = video_prefix + contexts
                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res
