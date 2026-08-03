from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import AutoModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..da3 import DA3DinoBackbone
from ..modules.qwen2 import Qwen2ForV3RCausalLM
from ..modules.qwen3 import Qwen3ForV3RCausalLM
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_internvl_chat import InternVLChatModel


class InternVL3RChatModel(InternVLChatModel):
    def __init__(
        self,
        config: InternVLChatConfig,
        vision_model=None,
        use_flash_attn=True,
    ):
        architecture: str = config.llm_config.architectures[0]
        if architecture == "Qwen2ForV3RCausalLM":
            language_model = Qwen2ForV3RCausalLM(config.llm_config)
        elif architecture == "Qwen3ForV3RCausalLM":
            language_model = Qwen3ForV3RCausalLM(config.llm_config)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        super().__init__(config, vision_model, language_model, use_flash_attn)

        self._build_condition_model(config)

        if config.vision_dpt_config is not None:
            architecture = config.vision_dpt_config.dpt_config.architectures[0]
            if architecture == "DA3Model":
                from ..da3 import DA3DepthHead

                dpt_head = DA3DepthHead(config.vision_dpt_config.dpt_config)
            else:
                raise ValueError(f"Unsupported dpt architecture: {architecture}")

            # Select the feature maps to be used
            dpt_head.register_forward_pre_hook(
                lambda _, inputs: (
                    [[inputs[0][i]] for i in config.vision_dpt_config.layer_ids],
                    int(0.5 * inputs[1]),  # Image width
                    int(0.5 * inputs[2]),  # Image height
                    *inputs[3:],
                )
            )
            setattr(self.language_model, "dpt_head", dpt_head)

    def _build_condition_model(self, config: InternVLChatConfig):
        self.condition_flag = False
        self.condition_model = None
        self.condition_start_idx = 1
        if config.vision_condition_config is not None:
            self.condition_flag = True
            cond_config = config.vision_condition_config
            architecture: str = cond_config.architectures[0]
            if architecture == "Dinov2Model":
                from transformers import Dinov2Model

                self.condition_model = Dinov2Model(cond_config)
            elif architecture == "DINOv3ViTModel":
                from transformers import DINOv3ViTModel

                self.condition_start_idx = 1 + 4  # cls_token + 4 register tokens
                self.condition_model = DINOv3ViTModel(cond_config)

                # Resize inputs, align to vit output tokens
                self.condition_model.register_forward_pre_hook(
                    lambda _, inputs: (
                        F.interpolate(inputs[0], scale_factor=16 / 14, mode="bilinear"),
                        *inputs[1:],
                    )
                )

            elif architecture == "DA3Model":
                self.condition_start_idx = 0
                self.condition_model = DA3DinoBackbone(cond_config)

            elif architecture == "VGGTModel":
                # from transformers import VGGTModel
                raise NotImplementedError("VGGTModel is not supported")
            else:
                raise ValueError(
                    f"Unsupported condition model architecture: {architecture}"
                )

            llm_hidden_size = config.llm_config.hidden_size
            self.condition_proj = nn.Linear(
                cond_config.hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            )
        else:
            self.condition_flag = (
                isinstance(self.select_layer, int) and self.select_layer > 0
            )
            if self.condition_flag:
                vit_hidden_size = config.vision_config.hidden_size
                llm_hidden_size = config.llm_config.hidden_size
                self.condition_proj = nn.Linear(
                    vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                    llm_hidden_size,
                )
            else:
                self.condition_proj = nn.Identity()

    def _merge_vit_tokens(self, tokens):
        h = w = int(tokens.shape[1] ** 0.5)
        tokens = tokens.reshape(tokens.shape[0], h, w, -1)
        tokens = self.pixel_shuffle(tokens, scale_factor=self.downsample_ratio)
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        return tokens

    def extract_feature(self, pixel_values, batch_size: int):
        # Get ViT mid/last features
        vit_outs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=self.condition_flag,
            return_dict=True,
        )

        vit_embeds = vit_outs.last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]
        vit_embeds = self._merge_vit_tokens(vit_embeds)
        vit_embeds = self.mlp1(vit_embeds)
        if not self.condition_flag:
            return vit_embeds, vit_embeds

        if self.condition_model is None:
            cond_embeds = vit_outs.hidden_states[self.select_layer]
        else:
            if isinstance(self.condition_model, DA3DinoBackbone):
                cond_embeds = self.condition_model(
                    rearrange(pixel_values, "(b s) c h w -> b s c h w", b=batch_size)
                ).last_hidden_state.flatten(0, 1)
            else:
                cond_embeds = self.condition_model(pixel_values).last_hidden_state

        cond_embeds = cond_embeds[:, self.condition_start_idx :, :]
        cond_embeds = self._merge_vit_tokens(cond_embeds)
        cond_embeds = self.condition_proj(cond_embeds)
        return cond_embeds, vit_embeds

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        mid_vit_embeds, vit_embeds = self.extract_feature(pixel_values, B)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size, _, vision_height, vision_width = pixel_values.shape

        frame_num = vit_batch_size // B

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = (
                input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]
            )

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            frame_num=frame_num,
            vision_embeds=mid_vit_embeds,
            visual_pos_masks=selected.view(B, N),
            vision_height=vision_height,
            vision_width=vision_width,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None

        # Prepare vision 3R task inputs
        frame_num = None
        vision_embeds = None
        visual_pos_masks = None
        vision_height = None
        vision_width = None

        if pixel_values is not None:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            vision_height, vision_width = pixel_values.shape[-2:]

            if visual_features is not None:
                vision_embeds = vit_embeds = visual_features
            else:
                vision_embeds, vit_embeds = self.extract_feature(pixel_values, B)
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)

            frame_num = vit_embeds.shape[0] // B
            visual_pos_masks = selected.view(B, N)

        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            frame_num=frame_num,
            vision_embeds=vision_embeds,
            visual_pos_masks=visual_pos_masks,
            vision_height=vision_height,
            vision_width=vision_width,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
