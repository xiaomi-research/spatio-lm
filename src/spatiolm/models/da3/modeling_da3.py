import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_da3 import DA3Config
from .dinov2.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)
from .modules.cam_dec import CameraDec
from .modules.cam_enc import CameraEnc
from .modules.dualdpt import DualDPT
from .utils.geometry import affine_inverse
from .utils.transform import pose_encoding_to_extri_intri


@dataclass
class DA3DinoBackboneOutput(BaseModelOutput):
    """
    Output class for DA3DinoBackbone.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_with_camera: Optional[
        List[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ] = None
    aux_features: Optional[List[torch.FloatTensor]] = None


class DA3DinoBackbone(nn.Module):
    """
    DA3Backbone class that wraps the DinoVisionTransformer for feature extraction.
    This provides the backbone for both depth and camera prediction tasks.
    Based on DepthAnything3Net's backbone extraction approach.
    """

    def __init__(self, config: DA3Config):
        super().__init__()
        self.config = config

        # Map config to DinoV2 model name
        model_name_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        self.name = config.backbone_name
        assert self.name in model_name_map, f"Unsupported backbone name {self.name}"

        model_cls = model_name_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.model = model_cls(
            img_size=config.image_size,
            patch_size=config.patch_size,
            ffn_layer=ffn_layer,
            alt_start=config.alt_start,
            qknorm_start=config.qknorm_start,
            rope_start=config.rope_start,
            cat_token=config.cat_token,
        )
        self.out_layers = config.out_layers

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> DA3DinoBackboneOutput:
        """
        Forward pass of the DA3 backbone.

        Args:
            pixel_values: Input tensor of shape (batch_size, num_channels, height, width)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights

        Returns:
            Tuple containing:
            - multi_level_features: List of tuples [(hidden_states, tokens), ...] for DPT processing
            - aux_features: Optional additional features for auxiliary tasks
        """
        # Get intermediate layers from DinoV2

        feats, aux_feats = self.model.get_intermediate_layers(
            pixel_values, self.out_layers, **kwargs
        )

        outputs = DA3DinoBackboneOutput(
            last_hidden_state=feats[-1][0],
            hidden_states_with_camera=feats,
            hidden_states=[f[0] for f in feats],
        )
        if len(aux_feats) > 0:
            outputs.aux_features = aux_feats

        return outputs


class DA3DepthHead(nn.Module):
    """
    Depth prediction head for DA3 model based on DPT architecture.
    This head takes multi-level features from the backbone and predicts depth maps.
    """

    _dynamic_tied_weights_keys = {
        f"decoder.scratch.output_conv2_aux.{i}.2.{wb}": f"decoder.scratch.output_conv2_aux.0.2.{wb}"
        for i in [1, 2, 3]
        for wb in ["weight", "bias"]
    }

    def __init__(self, config: DA3Config):
        super().__init__()
        self.config = config

        self.decoder = DualDPT(
            dim_in=config.depth_dim_in,
            output_dim=config.depth_dim_out,
            features=config.depth_features,
            out_channels=config.depth_out_channels,
        )

        self._patch_start_idx = 0

    def forward(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.decoder(feats, H, W, patch_start_idx=self._patch_start_idx)


class DA3CameraHead(nn.Module):
    """
    Camera parameter prediction head for DA3 model based on CameraDec architecture.
    This head takes features from the backbone and predicts camera parameters.
    """

    def __init__(self, config: DA3Config):
        super().__init__()
        self.config = config
        self.encoder = CameraEnc(dim_out=config.camera_enc_dim_out)
        self.decoder = CameraDec(dim_in=config.camera_dec_dim_in)

    def encode(self, ext: torch.Tensor, ixt: torch.Tensor, image_size: tuple):
        return self.encoder(ext, ixt, image_size)

    def decode(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        pose_enc = self.decoder(feats[-1][1])
        # Convert pose encoding to extrinsics and intrinsics
        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))

        outs = dict(extrinsics=affine_inverse(c2w), intrinsics=ixt)
        if outputs is not None:
            outputs.update(outs)
            return outputs
        return outs


class DA3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DA3Config
    base_model_prefix = "depth_anything_3"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DA3Model(DA3PreTrainedModel):
    """
    The complete DA3 model that combines backbone, depth head, and camera head.
    """

    def __init__(self, config: DA3Config):
        super().__init__(config)

        self.backbone = DA3DinoBackbone(config)
        self.depth_head = DA3DepthHead(config)
        self.camera_head = DA3CameraHead(config)

        # Initialize weights
        # self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: List[int] = [],
        output_raymaps: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        cam_token = None
        if extrinsics is not None:
            with torch.autocast(
                device_type=pixel_values.device.type,
                enabled=False,
            ):
                cam_token = self.camera_head.encode(
                    extrinsics,
                    intrinsics,
                    pixel_values.shape[-2:],
                )

        backbone_outs: DA3DinoBackboneOutput = self.backbone(
            pixel_values,
            cam_token=cam_token,
            export_feat_layers=export_feat_layers,
        )
        feats_with_cam = backbone_outs.hidden_states_with_camera
        # feats = [[item for item in feat] for feat in feats]
        H, W = pixel_values.shape[-2], pixel_values.shape[-1]

        # Process features through depth head
        with torch.autocast(
            device_type=pixel_values.device.type,
            enabled=False,
        ):
            depth_outs = self.depth_head(feats_with_cam, H, W)
            camera_outs = self.camera_head.decode(feats_with_cam, H, W)
            outputs = depth_outs | camera_outs

        # Unused outputs
        if not output_raymaps:
            outputs.pop("ray", None)
            outputs.pop("ray_conf", None)

        if output_hidden_states:
            outputs["hidden_states"] = backbone_outs.hidden_states

        return outputs
