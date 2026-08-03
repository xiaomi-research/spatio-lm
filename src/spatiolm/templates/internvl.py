import random
from typing import Any, Dict, List, Union

import numpy as np
import swift.llm.template.vision_utils as swift_vision_utils
import torch
from PIL import Image
from swift.llm.template.template.internvl import Internvl2Template, InternvlTemplate
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import findall
from swift.llm.template.vision_utils import _build_transform, _dynamic_preprocess
from swift.utils import get_env_args, is_deepspeed_enabled

from spatiolm.models import InternVL3RChatModel

from ..utils.action_mask import ActionMasker

__all__ = ["InternvlV3RTemplate", "InternvlVLA0Template"]


def transform_image(image, input_size=448, max_num=12, transform=None):
    if transform is None:
        transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_video_internvl(video: Union[str, bytes], bound=None, num_segments=32):
    from decord import VideoReader, cpu

    video_io = swift_vision_utils.load_file(video)
    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images = []
    frame_indices = swift_vision_utils._get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    # Ensure the first frame is included
    if len(frame_indices) > 0 and frame_indices[0] != 0:
        frame_indices[0] = 0

    for frame_index in frame_indices:
        images.append(Image.fromarray(vr[frame_index].asnumpy()).convert("RGB"))
    return images


# Monkey patch the load_video_internvl to ensure the first frame is included
swift_vision_utils.load_video_internvl = load_video_internvl


class InternvlV3RTemplate(Internvl2Template):
    def init_env_args(self):
        super().init_env_args()

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super(InternvlTemplate, self)._encode(inputs)
        input_ids = encoded["input_ids"]
        idx_list = findall(input_ids, -100)
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)
        images = inputs.images
        if images:
            has_video = bool(inputs.videos)
            if self.num_image_token is None:
                self.num_image_token = int((self.input_size // 14) ** 2 * (0.5**2))
            # max_num = self.max_num
            max_num = self.video_segments - 1  # align to video segments + thumbnail
            if has_video:
                max_num = self.video_max_num
            pixel_values = [
                transform_image(
                    image,
                    self.input_size,
                    max_num,
                    getattr(self, "transforms", None),
                )
                for image in images
            ]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model_info.torch_dtype)

            if pixel_values.shape[0] < self.video_segments:
                # assert len(num_patches) == 1, "Only Images are supported for now"
                if len(num_patches) == 1:
                    num_patches[0] = self.video_segments
                    repeat_num = self.video_segments - pixel_values.shape[0]
                    repeat_pixel_values = [pixel_values[-1:]] * repeat_num
                    pixel_values = torch.cat([pixel_values, *repeat_pixel_values])
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(idx_list), (
            f"len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}"
        )

        def _get_new_tokens(i):
            img_tokens: List[int] = (
                self.processor.encode("<IMG_CONTEXT>", add_special_tokens=False)
                * self.num_image_token
                * num_patches[i]
            )
            # TODO: 加上Camera的Special Tokens
            return img_tokens

        encoded["input_ids"], encoded["labels"], encoded["loss_scale"] = (
            self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens
            )
        )
        encoded["pixel_values"] = pixel_values
        encoded["num_patches"] = num_patches
        return encoded

    def _post_encode(self, model: InternVL3RChatModel, inputs):
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = inputs["input_ids"]
        inputs_embeds = embedding(input_ids).to(device=device)
        pixel_values = inputs.get("pixel_values")

        # Prepare vision 3R task inputs
        frame_num = None
        vision_embeds = None
        vision_pos_masks = None
        vision_height = None
        vision_width = None

        if pixel_values is not None:
            B, N, C = inputs_embeds.shape
            pixel_values = pixel_values.to(device=device)
            vision_embeds, vit_embeds = model.extract_feature(pixel_values, B)
            vision_embeds = vision_embeds.to(device=device)
            vit_embeds = vit_embeds.to(device=device)

            selected = (
                input_ids
                == self.processor.encode("<IMG_CONTEXT>", add_special_tokens=False)[0]
            )
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(
                dtype=inputs_embeds.dtype
            )

            frame_num = vit_embeds.shape[0] // B
            vision_pos_masks = selected.view(B, N)
            vision_height = pixel_values.shape[-2]
            vision_width = pixel_values.shape[-1]

        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros(
                (1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype
            )
            vision_embeds, vit_embeds = model.extract_feature(dummy_pixel_values, 1)
            vision_embeds = vision_embeds.to(device=device)
            vit_embeds = vit_embeds.to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.0
            vision_height = dummy_pixel_values.shape[-2]
            vision_width = dummy_pixel_values.shape[-1]

        return {
            "inputs_embeds": inputs_embeds,
            "frame_num": frame_num,
            "vision_embeds": vision_embeds,
            "visual_pos_masks": vision_pos_masks,
            "vision_height": vision_height,
            "vision_width": vision_width,
        }


class InternvlVLA0Template(InternvlV3RTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._build_transforms()

        self.action_masker = ActionMasker(self.tokenizer, mask_token="?")

    def _build_transforms(self):
        import torchvision.transforms as tvf
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transforms = tvf.Compose(
            [
                tvf.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                tvf.RandomResizedCrop(
                    (self.input_size, self.input_size),
                    scale=(0.7 * 0.7, 1.0),
                    # ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # tvf.CenterCrop(224),
                # tvf.Resize(
                #     (self.input_size, self.input_size),
                #     interpolation=InterpolationMode.BICUBIC,
                # ),
                tvf.ColorJitter(0.2, 0.2, 0.2, 0.05),
                tvf.ToTensor(),
                tvf.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def init_env_args(self):
        super().init_env_args()

        # Random Mask Action to '?'
        self.action_mask_proba = get_env_args("action_mask_proba", float, 0.9)
        self.action_mask_ratio = get_env_args("action_mask_ratio", float, 0.4)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        encoded = self.action_masker(
            encoded, self.action_mask_proba, self.action_mask_ratio
        )
        return encoded
