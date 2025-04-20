# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

from itertools import chain
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.rrdbnet_arch import RRDB


def get_sft(condition_in_channels,out_channels,sft_type):
    intermediate_channels = condition_in_channels // 4
    if sft_type == "SFT":
        return SFT(
            in_channels=condition_in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels
        )
    elif sft_type == "SFTResBlock":
        return SFTResBlock(
            in_channels=condition_in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels
        )
    else:
        print(f"unknown sft_type: {sft_type}")
        raise ValueError


class SFT(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels, 
            intermediate_channels=128,
            groups=32,
            eps=1e-6,
            ):
        super().__init__()
        self.out_channels = out_channels

        self.norm = InflatedGroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.mlp_shared = nn.Sequential(
            InflatedConv3d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )
        self.mlp_gamma = InflatedConv3d(intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.mlp_beta = InflatedConv3d(intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state, condition,):
        # hidden_state b c f h w
        assert hidden_state.size(1) == self.out_channels, "Input channel size mismatch"
        
        # InflatedGroupNorm  (b f) c h w
        hidden_state = self.norm(hidden_state)

        actv = self.mlp_shared(condition)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = hidden_state * (1 + gamma) + beta

        return out

class SFTResBlock(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels, 
            intermediate_channels=128,
            groups=32,
            eps=1e-6,
            ):
        super(SFTResBlock, self).__init__()
        self.sft1 = SFT(in_channels, out_channels, intermediate_channels, groups, eps)
        self.selu1 = nn.SELU(inplace=True)
        self.conv1 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.sft2 = SFT(in_channels, out_channels, intermediate_channels, groups, eps)
        self.selu2 = nn.SELU(inplace=True)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, condition):
        residual = x

        out = self.sft1(x, condition)
        out = self.selu1(out)
        out = self.conv1(out)

        out = self.sft2(out, condition)
        out = self.selu2(out)
        out = self.conv2(out)

        out += residual
        return out



class SKFF_Nonlinear(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF_Nonlinear, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.nonlinearity = nn.SELU()
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        # add nonlinearity here
        feats_V = self.norm(feats_V)
        feats_V = self.nonlinearity(feats_V)
        feats_V = self.conv_out(feats_V)

        return feats_V   
    

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)


        return feats_V   
    

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class WaveletEncoder(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)


        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)



        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    
    
class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
    
    
class FuseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = InflatedConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # zero init
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, hidden_states, fusion_hidden_states):
        return self.conv(fusion_hidden_states) + hidden_states

class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=False,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest"
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # if self.use_conv:
        #     if self.name == "conv":
        #         hidden_states = self.conv(hidden_states)
        #     else:
        #         hidden_states = self.Conv2d_0(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(
        self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = InflatedConv3d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_inflated_groupnorm=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        assert use_inflated_groupnorm != None
        if use_inflated_groupnorm:
            self.norm1 = InflatedGroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )

        self.conv1 = InflatedConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )

            self.time_emb_proj = torch.nn.Linear(
                temb_channels, time_emb_proj_out_channels
            )
        else:
            self.time_emb_proj = None

        if use_inflated_groupnorm:
            self.norm2 = InflatedGroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor



class ResnetBlock3D_Real(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = InflatedGroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )

            self.time_emb_proj = torch.nn.Linear(
                temb_channels, time_emb_proj_out_channels
            )
        else:
            self.time_emb_proj = None

        self.norm2 = InflatedGroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 初始化
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)


        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor



class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))
