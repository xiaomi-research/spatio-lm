from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from swift.trainers import Seq2SeqTrainer
from swift.utils import get_env_args

from ..losses import Distill3dLoss, DistillTokenLoss

__all__ = ["Seq2Seq3DTrainer"]


def clone_nested_tensor(obj):
    """
    Clone tensors in nested structures (dict, list, tuple).

    Args:
        obj: The object to clone. Can be a dict, list, tuple, or tensor.

    Returns:
        The same structure with all tensors cloned.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: clone_nested_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clone_nested_tensor(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clone_nested_tensor(item) for item in obj)
    else:
        return obj


class Seq2Seq3DTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        assert "teacher3d" in kwargs, "teacher3d must be provided"
        self.teacher3d: nn.Module = kwargs.pop("teacher3d").eval()
        self.teacher3d_input_downsample = 0.5

        self.loss_token = DistillTokenLoss()
        self.loss_3d = Distill3dLoss()

        super().__init__(*args, **kwargs)

        self._pred_token_distill_ids = self.model.config.vision_dpt_config.layer_ids

        self._lm_weight = get_env_args("loss_lm_weight", float, 0.6)
        self._token_weight = get_env_args("loss_token_weight", float, 0.2)
        self._depth_weight = get_env_args("loss_depth_weight", float, 0.1)
        self._ray_weight = get_env_args("loss_ray_weight", float, 0.1)

        self._dpt_lr_scale = get_env_args("dpt_lr_scale", float, 10.0)

        self._loss_logs = dict()

    def infer_teacher3d(self, pixel_values, batch_size):
        # Downsample pixel values for 3D model input
        pixel_values = F.interpolate(
            pixel_values,
            scale_factor=self.teacher3d_input_downsample,
            mode="bilinear",
        )

        with torch.inference_mode(), torch.autocast(
            device_type=pixel_values.device.type,
            dtype=pixel_values.dtype,
        ):
            # 3D outputs
            outs_3d = self.teacher3d(
                rearrange(
                    pixel_values,
                    "(b s) c h w -> b s c h w",
                    b=batch_size,
                ).float(),
                output_raymaps=True,
                output_hidden_states=True,
            )

        # Clone outputs, used for loss computation
        outs_3d = clone_nested_tensor(outs_3d)

        return outs_3d

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        batch_size = inputs["input_ids"].shape[0]

        # Enable output hidden states for 3D model
        inputs["output_hidden_states"] = True
        loss_lm, model_outs = super().compute_loss(
            model, inputs, True, num_items_in_batch
        )

        ref_outs_3d = self.infer_teacher3d(inputs["pixel_values"], batch_size)

        loss_token = self.loss_token(
            [model_outs["condition_3d"][i] for i in self._pred_token_distill_ids],
            ref_outs_3d["hidden_states"],
        )
        loss_3d = self.loss_3d(model_outs.dpt_3d, ref_outs_3d)
        loss = (
            loss_lm * self._lm_weight
            + loss_token * self._token_weight
            + loss_3d["depth"] * self._depth_weight
            + loss_3d["ray"] * self._ray_weight
        )

        self._loss_logs = {
            "loss_lm": loss_lm.detach(),
            "loss_token": loss_token.detach(),
            "loss_depth": loss_3d["depth"].detach(),
            "loss_ray": loss_3d["ray"].detach(),
        }

        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        aux_logs = {
            k: self.accelerator.gather_for_metrics(v).mean().item()
            * self.args.gradient_accumulation_steps
            for k, v in self._loss_logs.items()
        }

        new_logs = {}
        if "loss" in logs:
            new_logs["loss"] = logs.pop("loss")
        new_logs.update(aux_logs)
        new_logs.update(logs)

        return super().log(new_logs, *args, **kwargs)

    def create_optimizer(self):
        self.optimizer = super().create_optimizer()

        dpt_head_param_ids = set()
        dpt_head = getattr(
            getattr(self.model, "language_model", None),
            "dpt_head",
            None,
        )
        if dpt_head is None:
            return self.optimizer

        for param in dpt_head.parameters():
            dpt_head_param_ids.add(id(param))

        # Remove dpt_head parameters
        for param_group in self.optimizer.param_groups:
            new_params = []
            for param in param_group["params"]:
                if id(param) not in dpt_head_param_ids:
                    new_params.append(param)
            param_group["params"] = new_params

        # Add dpt_head parameters with scaled lr
        self.optimizer.add_param_group(
            {
                "params": [p for p in dpt_head.parameters() if p.requires_grad],
                "lr": self.args.learning_rate * self._dpt_lr_scale,
            }
        )

        return self.optimizer
