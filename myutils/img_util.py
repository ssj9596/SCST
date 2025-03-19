import math
import torch
import torchvision
import imageio
from basicsr.archs.arch_util import flow_warp

from einops import rearrange
import torch.nn.functional as F

def save_videos_grid(videos, path=None, rescale=True, n_rows=4, fps=8, discardN=0):
    videos = rearrange(videos, "b c t h w -> t b c h w").cpu()
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x / 2.0 + 0.5).clamp(0, 1)  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        #x = adjust_gamma(x, 0.5)
        outputs.append(x)

    outputs = outputs[discardN:]

    if path is not None:
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, outputs, duration=1000/fps, loop=0)

    return outputs

def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image

def colorful_loss(pred):
    colorfulness_loss = 0
    for i in range(pred.shape[0]):
        (R, G, B) = pred[i][0], pred[i][1], pred[i][2]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R+G) - B)
        (rbMean, rbStd) = (torch.mean(rg), torch.std(rg))
        (ybMean, ybStd) = (torch.mean(yb), torch.std(yb))
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = stdRoot + (0.3 * meanRoot)
        colorfulness_loss += (1 - colorfulness)
    return colorfulness_loss

def compute_temporal_condition(flows, latents, masks,num_frame):
    # flow_f: [b,t-1,2,h,w], (backward flow, for forward propagation)
    # flow_b: [b,t-1,2,h,w], (forward flow, for backward propagation)
    flow_fwd_prop, flow_bwd_prop = flows
    fwd_occs, bwd_occs = masks
    t = num_frame
    # fwd_occ_list, bwd_occ_list = list(), list()
    # for i in range(t-1):
    #     fwd_flow, bwd_flow = flow_bwd_prop[:, i, :, :, :], flow_fwd_prop[:, i, :, :, :]
    #     fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
    #     fwd_occ_list.append(fwd_occ)
    #     bwd_occ_list.append(bwd_occ)

    # b,t,c,h//8,w//8
    latents = rearrange(latents, '(b t) c h w -> b t c h w', t=t)
    # compute the forward loss and backward loss
    loss_b = 0
    latent_curr_warp = torch.zeros_like(latents[:, -1, :, :, :])
    # backward propagation
    for i in range(t - 1, -1, -1):
        latent_curr = latents[:, i, :, :, :]
        if i < t - 1:  # no warping required for the last timestep
            flow = flow_bwd_prop[:, i, :, :, :]
            latent_curr_warp = flow_warp(latent_curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_b += F.l1_loss((1 - fwd_occs[:, i, :, :, :]) * latent_prev, (1 - fwd_occs[:, i, :, :, :]) * latent_curr, reduction='sum')
        latent_prev = latent_curr_warp
    loss_f = 0
    latent_curr_warp = torch.zeros_like(latents[:, 0, :, :, :])
    # forward propagation
    for i in range(0, t):
        latent_curr = latents[:, i, :, :, :]
        if i > 0:  # no warping required for the first timestep
            flow = flow_fwd_prop[:, i - 1, :, :, :]
            latent_curr_warp = flow_warp(latent_curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_f += F.l1_loss((1 - bwd_occs[:, i-1, :, :, :]) * latent_prev, (1 - bwd_occs[:, i-1, :, :, :]) * latent_curr, reduction='sum')
        latent_prev = latent_curr_warp
    return loss_b + loss_f