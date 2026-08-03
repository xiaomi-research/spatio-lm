from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ACT2FN,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import check_model_inputs

from ..utils import make_vision_attn_mask
from .utils import CausalLM3dOutputWithPast


class Qwen3V3RConfig(Qwen3Config):
    model_type = "qwen3_v3r"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0,
        # For vision 3r config
        vision3r_layer_ids=(4, 11, 17),
        vision3r_intermediate_size=None,
        vision3r_head_dim=None,
        vision3r_num_attention_heads=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            layer_types,
            attention_dropout,
            **kwargs,
        )

        # 添加一些3R相关的配置
        self.vision3r_layer_ids = vision3r_layer_ids
        self.vision3r_intermediate_size = (
            vision3r_intermediate_size or intermediate_size
        )
        self.vision3r_num_attention_heads = (
            vision3r_num_attention_heads or num_attention_heads
        )
        self.vision3r_head_dim = vision3r_head_dim or head_dim


class Qwen3Vision3RLayer(nn.Module):
    def __init__(self, config: Qwen3V3RConfig, layer_idx: int):
        super().__init__()

        self.in_zero_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        self.out_zero_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        self.frame_layer = Qwen3DecoderLayer(config, layer_idx)
        self.frame_layer.self_attn.is_causal = False

        self.global_layer = Qwen3DecoderLayer(config, layer_idx)
        self.global_layer.self_attn.is_causal = False

    def forward(
        self,
        cond_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        frame_attn_mask: torch.Tensor,
        global_attn_mask: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        cond_embeds = self.in_zero_proj(cond_embeds)
        hidden_states = hidden_states + cond_embeds

        # Frame attention
        hidden_states = self.frame_layer(
            hidden_states,
            attention_mask=frame_attn_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # Global attention
        hidden_states = self.global_layer(
            hidden_states,
            attention_mask=global_attn_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        return hidden_states, self.out_zero_proj(hidden_states)


class Qwen3Vision3RCondition(nn.Module):
    def __init__(self, config: Qwen3V3RConfig):
        super().__init__()
        config = deepcopy(config)

        # FIXME: "sdpa" 的实现有点问题
        config._attn_implementation = "eager"
        # Setup config for Attention layers
        config.intermediate_size = config.vision3r_intermediate_size
        config.num_attention_heads = config.vision3r_num_attention_heads
        config.head_dim = config.vision3r_head_dim

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        # For 3R, we need to modify the attention mechanism
        # self.layer_ids = getattr(config, "vision3r_layer_ids", [4, 11, 17])
        self.layer_ids = config.vision3r_layer_ids

        self.layers = nn.ModuleList(
            [Qwen3Vision3RLayer(config, idx) for idx in self.layer_ids]
        )

    def get_layer(self, idx: int):
        if idx not in self.layer_ids:
            return False
        return self.layers[self.layer_ids.index(idx)]

    def prepare_pos_attn_mask(self, frame_nums: int, cond_embeds: torch.Tensor):
        token_num = cond_embeds.shape[1]

        assert token_num % frame_nums == 0, (
            "The number of frames must divide the number of tokens evenly."
        )

        position_ids = torch.arange(cond_embeds.shape[1], device=cond_embeds.device)
        position_embeddings = self.rotary_emb(cond_embeds, position_ids.unsqueeze(0))

        frame_attn_mask = make_vision_attn_mask(
            token_num,
            frame_nums,
            device=cond_embeds.device,
            dtype=cond_embeds.dtype,
            attn_type="frame",
        ).contiguous()
        global_attn_mask = make_vision_attn_mask(
            token_num,
            frame_nums,
            device=cond_embeds.device,
            dtype=cond_embeds.dtype,
            attn_type="global",
        ).contiguous()

        return {
            "position_embeddings": position_embeddings,
            "frame_attn_mask": frame_attn_mask,
            "global_attn_mask": global_attn_mask,
        }


# @auto_docstring
class Qwen3V3RModel(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.vision_3r_cond = Qwen3Vision3RCondition(config)

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        frame_num: Optional[int] = None,
        vision_embeds: Optional[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        output_cond_hidden_states: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        batch_size, _, hidden_dim = hidden_states.shape

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Prepare params for condition layer
        last_cond = vision_embeds.view(batch_size, -1, hidden_dim)
        condition_kwargs = self.vision_3r_cond.prepare_pos_attn_mask(
            frame_num, last_cond
        )

        # Get condition control outputs
        condition_outputs = dict()
        condition_hidden_states = []
        for idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            condition_layer = self.vision_3r_cond.get_layer(idx)
            if condition_layer and (
                past_key_values is None or past_key_values[idx][0] is None
            ):
                last_cond, last_cond_proj = condition_layer(
                    # Condition, such as vision_embeds, geometry tokens, etc.
                    last_cond,
                    # Vision parts of Hidden states from VLM
                    hidden_states[visual_pos_masks, :]
                    .view(batch_size, -1, hidden_dim)
                    .contiguous(),
                    **condition_kwargs,
                )
                condition_hidden_states.append(
                    last_cond.view(batch_size, frame_num, -1, hidden_dim)
                )
                condition_outputs[str(idx)] = last_cond_proj

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if str(idx) in condition_outputs:
                hidden_states[visual_pos_masks, :] = (
                    condition_outputs[str(idx)].flatten(0, 1)
                    + hidden_states[visual_pos_masks, :]
                )

        if len(condition_outputs) > 0:
            condition_hidden_states.append(
                hidden_states[visual_pos_masks, :].view(
                    batch_size, frame_num, -1, hidden_dim
                )
            )

        if not output_cond_hidden_states or len(condition_hidden_states) == 0:
            condition_hidden_states = None

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=condition_hidden_states,
        )


# @auto_docstring
class Qwen3ForV3RCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3V3RModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, modules):
        super()._init_weights(modules)
        # 对没加载上的参数使用你自己的初始化方法
        for name, param in modules.named_parameters():
            if param.requires_grad and param.data.numel() > 0:
                if "zero_proj" in name:
                    nn.init.constant_(param.data, 0.0)
                elif "fuse_proj" in name:
                    nn.init.normal_(param.data)

    @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # For Vision 3R Task
        frame_num: Optional[int] = None,
        vision_embeds: Optional[torch.FloatTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        vision_height: Optional[int] = None,
        vision_width: Optional[int] = None,
        vision_3d_outputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            frame_num=frame_num,
            vision_embeds=vision_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            output_cond_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        dpt_outs = None
        if outputs.hidden_states is not None:
            setattr(self, "condition_3d", outputs.hidden_states)
            if hasattr(self, "dpt_head"):
                with torch.autocast(
                    device_type=hidden_states.device.type,
                    dtype=torch.float32,
                ):
                    dpt_outs = self.dpt_head(
                        outputs.hidden_states, vision_height, vision_width
                    )
                    if vision_3d_outputs is not None:
                        vision_3d_outputs.update(dpt_outs)

        return CausalLM3dOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            condition_3d=outputs.hidden_states,
            dpt_3d=dpt_outs,
        )


__all__ = ["Qwen3V3RConfig", "Qwen3V3RModel", "Qwen3ForV3RCausalLM"]
