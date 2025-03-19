# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py

import pdb

from einops import rearrange
import torch
from torch import nn

from .motion_module import get_motion_module
from .resnet import Downsample3D, ResnetBlock3D, Upsample3D, get_sft
from .transformer_3d import Transformer3DModel


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    name_index=None,
    use_sft=None,
    sft_type=None,
):
    # print(sft_type)
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            use_sft=use_sft,
            sft_type=sft_type
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock3D"
            )
        if name_index is not None:
            name_index = f"CrossAttnDownBlock_{name_index}_"
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            name=name_index,
            use_sft=use_sft,
            sft_type=sft_type
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    unet_use_cross_frame_attention=None,
    unet_use_temporal_attention=None,
    use_inflated_groupnorm=None,
    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
    name_index=None,
    use_sft=None,
    sft_type=None,
    use_controlnet_guide_attention=False,
    controlnet_guide_attention_temb=False,
    motion_module_after_cg_attn=False
    


):
    # print(sft_type)
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            use_sft=use_sft,
            sft_type=sft_type,
            use_controlnet_guide_attention=use_controlnet_guide_attention,
            controlnet_guide_attention_temb=controlnet_guide_attention_temb,
            attn_num_head_channels=attn_num_head_channels,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            motion_module_after_cg_attn=motion_module_after_cg_attn




        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock3D"
            )
        if name_index is not None:
            name_index = f"CrossAttnUpBlock_{name_index}_"
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            name=name_index,
            use_sft=use_sft,
            sft_type=sft_type,
            use_controlnet_guide_attention=use_controlnet_guide_attention,
            controlnet_guide_attention_temb=controlnet_guide_attention_temb,
            motion_module_after_cg_attn=motion_module_after_cg_attn

        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        name=None,
        use_sft=None,
        sft_type=None,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.name = name
        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        ]

        sft_modules = [get_sft(
            condition_in_channels=in_channels,
            out_channels=in_channels,
            sft_type=sft_type
        ) if use_sft else None]

        attentions = []
        motion_modules = []

        for i in range(num_layers):


            sft_modules.append(
                get_sft(
                    condition_in_channels=in_channels,
                    out_channels=in_channels,
                    sft_type=sft_type
                )
                if use_sft else None
            )



            if dual_cross_attention:
                raise NotImplementedError
            if self.name is not None:
                attn_name = f"{self.name}_{i}_TransformerModel"
            else:
                attn_name = None
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    name=attn_name,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=in_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        self.sft_modules = nn.ModuleList(sft_modules)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        self_attention_additional_feats=None,
        mode=None,
        controlnet_conditions=None,
    ):
        # resnet first
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = (
            self.sft_modules[0](hidden_states,controlnet_conditions)
            if self.sft_modules[0] is not None
            else hidden_states
        )

        for sft, attn, resnet, motion_module in zip(
            self.sft_modules[1:], self.attentions, self.resnets[1:], self.motion_modules
        ):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_additional_feats=self_attention_additional_feats,
                mode=mode,
            ).sample
            hidden_states = (
                motion_module(
                    hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                )
                if motion_module is not None
                else hidden_states
            )
            hidden_states = resnet(hidden_states, temb)
            hidden_states = (
                sft(hidden_states,controlnet_conditions)
                if sft is not None
                else hidden_states
            )

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        name=None,
        use_sft=None,
        sft_type=None,
        
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []
        sft_modules = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.name=name

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
    
            sft_modules.append(
                get_sft(
                    condition_in_channels=out_channels,
                    out_channels=out_channels,
                    sft_type=sft_type
                )
                if use_sft else None
            )


            if dual_cross_attention:
                raise NotImplementedError
            if self.name is not None:
                attn_name = f"{self.name}_{i}_TransformerModel"
            else:
                attn_name = None
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    name=attn_name,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        self.sft_modules = nn.ModuleList(sft_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        self_attention_additional_feats=None,
        mode=None,
        controlnet_conditions=None,
    ):
        output_states = ()

        for i, (resnet, sft, attn, motion_module) in enumerate(
            zip(self.resnets, self.sft_modules, self.attentions, self.motion_modules)
        ):
            controlnet_condition = controlnet_conditions[0]
            controlnet_conditions = controlnet_conditions[1:]
            # self.gradient_checkpointing = False
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    self_attention_additional_feats,
                    mode,
                )[0]

                if sft is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(sft),
                        hidden_states,
                        controlnet_condition,
                    )

                # add motion module
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )

            else:
        
                hidden_states = resnet(hidden_states, temb)



                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    self_attention_additional_feats=self_attention_additional_feats,
                    mode=mode,
                ).sample

                hidden_states = (
                    sft(hidden_states,controlnet_condition)
                    if sft is not None
                    else hidden_states
                )

                # add motion module
                hidden_states = (
                    motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        use_sft=None,
        sft_type=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []
        sft_modules = []

        # use_motion_module = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            sft_modules.append(
                get_sft(
                    condition_in_channels=out_channels,
                    out_channels=out_channels,
                    sft_type=sft_type
                )
                if use_sft else None
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        self.sft_modules = nn.ModuleList(sft_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, controlnet_conditions=None):
        output_states = ()

        for i, (resnet,sft, motion_module) in enumerate(zip(self.resnets, self.sft_modules, self.motion_modules)):
            controlnet_condition = controlnet_conditions[0]
            controlnet_conditions = controlnet_conditions[1:]

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                if sft is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(sft),
                        hidden_states,
                        controlnet_condition,
                    )
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = (
                    sft(hidden_states,controlnet_condition)
                    if sft is not None
                    else hidden_states
                )
                # add motion module
                hidden_states = (
                    motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_motion_module=None,
        use_inflated_groupnorm=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        name=None,
        use_sft=None,
        sft_type=None,
        use_controlnet_guide_attention=False,
        controlnet_guide_attention_temb=False,
        motion_module_after_cg_attn = False
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []
        sft_modules = []
        controlnet_guide_attentions = []


        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.name = name
        self.motion_module_after_cg_attn = motion_module_after_cg_attn

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            sft_modules.append(
                get_sft(
                    condition_in_channels=res_skip_channels,
                    out_channels=out_channels,
                    sft_type=sft_type
                )
                if use_sft else None
            )   

            if dual_cross_attention:
                raise NotImplementedError
            if self.name is not None:
                attn_name = f"{self.name}_{i}_TransformerModel"
            else:
                attn_name = None
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    name=attn_name,
                )
            )
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )
            # pixel attention
            if use_controlnet_guide_attention:
                controlnet_guide_attentions.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=res_skip_channels, #### control net see663
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        name=None,
                        temb_channels=temb_channels if controlnet_guide_attention_temb else None,
                    )
                )
            else:
                controlnet_guide_attentions.append(None)


        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        self.sft_modules = nn.ModuleList(sft_modules)
        self.controlnet_guide_attentions = nn.ModuleList(controlnet_guide_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        self_attention_additional_feats=None,
        mode=None,
        controlnet_conditions=None,
        control_temb=None,
    ):
        for i, (resnet, sft, attn, motion_module, pix_attn) in enumerate(
            zip(self.resnets, self.sft_modules, self.attentions, self.motion_modules, self.controlnet_guide_attentions)
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            controlnet_condition = controlnet_conditions[-1]
            controlnet_conditions = controlnet_conditions[:-1]
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_additional_feats=self_attention_additional_feats,
                mode=mode,
            ).sample

            hidden_states = (
                motion_module(
                    hidden_states, temb=None, encoder_hidden_states=encoder_hidden_states
                )
                if motion_module is not None and not self.motion_module_after_cg_attn
                else hidden_states
            )
            # add sft
            hidden_states = (
                # upblock 反过来
                sft(hidden_states,controlnet_condition)
                if sft is not None
                else hidden_states
            )

            # pixel attention
            if pix_attn is not None:
                hidden_states = pix_attn(
                    hidden_states,
                    encoder_hidden_states=rearrange(controlnet_condition, 'b c f h w  -> (b f) (h w) c'),
                    self_attention_additional_feats=self_attention_additional_feats,
                    return_dict=False,
                    control_temb=control_temb,
                )


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        use_inflated_groupnorm=None,
        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
        use_sft=None,
        sft_type=None,
        use_controlnet_guide_attention=False,
        controlnet_guide_attention_temb=False,
        attn_num_head_channels=1,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        motion_module_after_cg_attn=False

    ):
        super().__init__()
        resnets = []
        motion_modules = []
        sft_modules = []
        controlnet_guide_attentions = []
        self.motion_module_after_cg_attn = motion_module_after_cg_attn
        print("UpBlock3D : self.motion_module_after_cg_attn",self.motion_module_after_cg_attn)
        # use_motion_module = False
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            sft_modules.append(
                get_sft(
                    condition_in_channels=res_skip_channels,
                    out_channels=out_channels,
                    sft_type=sft_type
                )
                if use_sft else None
            )

            # after
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
                if use_motion_module
                else None
            )

            if use_controlnet_guide_attention:
                controlnet_guide_attentions.append(
                    Transformer3DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=res_skip_channels, #### control net see663
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                        name=None,
                        temb_channels=temb_channels if controlnet_guide_attention_temb else None,
                    )
                )
            else:
                controlnet_guide_attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)
        self.sft_modules = nn.ModuleList(sft_modules)
        self.controlnet_guide_attentions = nn.ModuleList(controlnet_guide_attentions)


        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        upsample_size=None,
        encoder_hidden_states=None,
        controlnet_conditions=None,
        control_temb=None,
    ):
        for i, (resnet, sft, motion_module,pix_attn) in enumerate(
            zip(self.resnets, self.sft_modules, self.motion_modules,self.controlnet_guide_attentions)
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            controlnet_condition = controlnet_conditions[-1]
            controlnet_conditions = controlnet_conditions[:-1]


            hidden_states = resnet(hidden_states, temb)
            hidden_states = (
                motion_module(
                    hidden_states, temb=None, encoder_hidden_states=encoder_hidden_states
                )
                if motion_module is not None
                else hidden_states
            )

            # sft
            hidden_states = (
                sft(hidden_states,controlnet_condition)
                if sft is not None
                else hidden_states
            )

            # pixel attention
            if pix_attn is not None:
                hidden_states = pix_attn(
                    hidden_states,
                    encoder_hidden_states=rearrange(controlnet_condition, 'b c f h w  -> (b f) (h w) c'),
                    return_dict=False,
                    control_temb=control_temb,
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
