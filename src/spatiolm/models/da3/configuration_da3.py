from typing import List, Literal

from transformers.configuration_utils import PretrainedConfig


class DA3Config(PretrainedConfig):
    """
    Configuration class for DA3 (Depth Anything 3) model.

    This configuration inherits from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_name (`str`, *optional*, defaults to `"facebook/dinov2-base"`):
            Name of the backbone model to use.
        backbone_config (`dict`, *optional*):
            Configuration for the backbone model if not using a pretrained model.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embeddings.
        use_rotary_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use rotary position embeddings.
        crop_size (`int`, *optional*, defaults to 224):
            The size to crop images to.
    """

    model_type = "depth_anything_3"

    def __init__(
        self,
        backbone_name: Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
        image_size: int = 518,
        patch_size: int = 14,
        out_layers: List[int] = [11, 15, 19, 23],
        alt_start: int = 8,
        qknorm_start: int = 8,
        rope_start: int = 8,
        cat_token: bool = True,
        # Depth head specific parameters
        depth_dim_in: int = 2048,
        depth_dim_out: int = 2,
        depth_features: int = 256,
        depth_out_channels: List[int] = [256, 512, 1024, 1024],
        # Camera head specific parameters
        camera_enc_dim_out: int = 1024,
        camera_dec_dim_in: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token

        self.hidden_size = depth_dim_in

        # Depth head config
        self.depth_dim_in = depth_dim_in
        self.depth_dim_out = depth_dim_out
        self.depth_features = depth_features
        self.depth_out_channels = depth_out_channels

        # Camera head config
        self.camera_enc_dim_out = camera_enc_dim_out
        self.camera_dec_dim_in = camera_dec_dim_in
