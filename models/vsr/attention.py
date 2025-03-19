# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from typing import Any, Dict, Optional

import torch
from diffusers.models.attention import AdaLayerNorm, Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from einops import rearrange
from torch import nn
from diffusers.utils.import_utils import is_xformers_available
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if not double_self_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if not self.use_ada_layer_norm_single:
            self.norm3 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    # important
    # def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: None):
    #     if not is_xformers_available():
    #         print("Here is how to install it")
    #         raise ModuleNotFoundError(
    #             "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
    #             " xformers",
    #             name="xformers",
    #         )
    #     elif not torch.cuda.is_available():
    #         raise ValueError(
    #             "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
    #             " available for GPU "
    #         )
    #     else:
    #         try:
    #             # Make sure we can run the memory efficient attention
    #             _ = xformers.ops.memory_efficient_attention(
    #                 torch.randn((1, 2, 40), device="cuda"),
    #                 torch.randn((1, 2, 40), device="cuda"),
    #                 torch.randn((1, 2, 40), device="cuda"),
    #             )
    #         except Exception as e:
    #             raise e
    #         self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
    #         if self.attn2 is not None:
    #             self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0)
            if cross_attention_kwargs is not None
            else 1.0
        )

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        name=None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.name=name

        # SC-Attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim)
        )

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False
        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        video_length=None,
        self_attention_additional_feats=None,
        mode=None,
        height=None
    ):
        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )
        if self.name:
            modify_norm_hidden_states = norm_hidden_states
            if mode == "write":
                self_attention_additional_feats[self.name]=norm_hidden_states
            elif mode == "read" and self_attention_additional_feats:
                ref_states = self_attention_additional_feats[self.name]
                bank_fea = [
                    rearrange(
                        ref_states.unsqueeze(1).repeat(1, video_length, 1, 1),
                        "b t l c -> (b t) l c",
                    )
                ]
                modify_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + bank_fea, dim=1
                )

            if self.unet_use_cross_frame_attention:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=modify_norm_hidden_states,
                        video_length=video_length,
                    )
                    + hidden_states
                )
            else:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states, 
                        encoder_hidden_states=modify_norm_hidden_states,
                        attention_mask=attention_mask
                    )
                    + hidden_states
                )
        else:    
            if self.unet_use_cross_frame_attention:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        video_length=video_length,
                    )
                    + hidden_states
                )
            else:
                hidden_states = (
                    self.attn1(norm_hidden_states, attention_mask=attention_mask)
                    + hidden_states
                )

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states




def pad_to_multiple(x, h, p):
    b, f, d, c = x.shape
    w = d // h

    # 计算需要填充的高度和宽度
    pad_h = (p - (h % p)) % p
    pad_w = (p - (w % p)) % p

    # 填充
    x = torch.nn.functional.pad(x.view(b, f, h, w, c), (0, 0, 0, pad_w, 0, pad_h))
    return x



def split_into_patches(x, h, p):
    # x 的形状为 (b, f, h*w, c)
    b, f, d, c = x.shape
    w = d // h
    # 填充到最近的 p 的倍数
    # print("======PAD=====")
    x = pad_to_multiple(x,h,p)
    # print(x)

    # 重新调整形状为 (b, f, h, w, c)
    new_h = x.shape[2]
    new_w = x.shape[3]
    x = x.view(b, f, new_h, new_w, c)

    # 使用 unfold 进行分块
    patches = x.unfold(2, p, p).unfold(3, p, p)  # 形状变为 (b, f, h/p, w/p, p, p, c)
    patches = patches.contiguous().view(b, f, new_h // p, new_w // p, p * p, c)  # 形状变为 (b, f, h/p, w/p, p*p, c)
    patches = rearrange(patches,"b f h w pp c -> (b h w) (pp f) c").contiguous()
    return patches,new_h,new_w


def merge_patches(patches, new_h,new_w, orig_h, orig_w, p):
    patches = rearrange(patches,"(b h w) (pp f) c -> b f h w pp c",h=new_h//p,w=new_w//p,pp=p*p)
    merged = patches.view(patches.shape[0], patches.shape[1], new_h // p, new_w // p, p, p, -1)  # 形状变为 (b, f, h/p, w/p, p, p, c)
    # 还原形状为 (b, f, h, w, c)
    merged = merged.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(patches.shape[0], patches.shape[1], new_h, new_w, -1)  # 还原为 (b, f, h, w, c)
    merged = merged[:, :, :orig_h, :orig_w, :]  # 只保留原始的 h 和 w 大小的部分
    #bf hw c
    merged = merged.reshape(patches.shape[0]*patches.shape[1], orig_h * orig_w, -1).contiguous()  # 还原为 (b, f, h*w, c)
    return merged


class TemporalCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        name=None,
        patch_size=0
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.name=name
        self.patch_size=patch_size

        # Cross-Temp-Attn
        if cross_attention_dim is not None:
            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            self.norm1 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.attn1 = None
            self.norm1 = None

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False
        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        video_length=None,
        self_attention_additional_feats=None,
        mode=None,
        height=None
    ):
        
        d = hidden_states.shape[1]
        if self.patch_size > 0:
            # (b f) d c -> (b h/p w/p) (pp f) c
            hidden_states,new_h,new_w = split_into_patches(
                                rearrange(hidden_states,"(b f) d c->b f d c",f=video_length),
                                height,self.patch_size)
        
        else:
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )
        
        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )
        if self.name:
            modify_norm_hidden_states = norm_hidden_states
            if mode == "write":
                self_attention_additional_feats[self.name]=norm_hidden_states
            elif mode == "read" and self_attention_additional_feats:
                ref_states = self_attention_additional_feats[self.name]
                bank_fea = [
                    rearrange(
                        ref_states.unsqueeze(1).repeat(1, video_length, 1, 1),
                        "b t l c -> (b t) l c",
                    )
                ]
                modify_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + bank_fea, dim=1
                )

            if self.unet_use_cross_frame_attention:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=modify_norm_hidden_states,
                        video_length=video_length,
                    )
                    + hidden_states
                )
            else:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states, 
                        encoder_hidden_states=modify_norm_hidden_states,
                        attention_mask=attention_mask
                    )
                    + hidden_states
                )
        else:    
            if self.unet_use_cross_frame_attention:
                hidden_states = (
                    self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        video_length=video_length,
                    )
                    + hidden_states
                )
            else:
                #Cross-Temporal-Attention
                # import pdb
                # pdb.set_trace()
                # print("TemporalCrossTransformerBlock!!!", hidden_states.shape)
                # print("motion_module hidden_states", hidden_states.shape)
                # print("motion_module encoder_hidden_states", encoder_hidden_states.shape)
                if self.patch_size > 0:
                    hidden_states = (
                        self.attn1(norm_hidden_states,
                                encoder_hidden_states=split_into_patches(
                                        rearrange(encoder_hidden_states,"(b f) d c->b f d c",f=video_length),
                                        height,self.patch_size)[0],
                                attention_mask=None)
                        + hidden_states
                    )
                else:
                    hidden_states = (
                        self.attn1(norm_hidden_states,
                                encoder_hidden_states=rearrange(
                                    #Temporal encoder_hidden_states
                                        encoder_hidden_states, "(b f) d c -> (b d) f c", f=video_length
                                        ),
                                attention_mask=None)
                        + hidden_states
                    )

        
        # Spatial
        if self.patch_size > 0:
            hidden_states = merge_patches(hidden_states, new_h,new_w, 
                                          height, d//height, self.patch_size)
        else: 
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
