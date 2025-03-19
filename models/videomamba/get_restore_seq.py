from einops import rearrange
import torch



def Continue_FHW(hidden_state):
    b, c, f, h, w = hidden_state.shape
    indices = torch.arange(h * w).reshape(h, w)
    

    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    

    hidden_state = rearrange(hidden_state,"b c f h w -> b c f (h w)")
    hidden_state = hidden_state[:, :, :, indices]
    

    frame_indices = torch.arange(f * h * w).reshape(f, h * w)
    frame_indices[1::2,:] = frame_indices[1::2,:].flip(-1)
    frame_indices = frame_indices.flatten()
    
    hidden_state = rearrange(hidden_state,"b c f n -> b (f n) c")
    hidden_state = hidden_state[:, frame_indices, :]
    # b (f h w) c
    return hidden_state.contiguous()


def Restore_FHW(hidden_state, f, h, w):

    frame_indices = torch.arange(f * h * w).reshape(f, h * w)
    frame_indices[1::2,:] = frame_indices[1::2,:].flip(-1)
    frame_indices = frame_indices.flatten()
    hidden_state = hidden_state[:, frame_indices, :]
    hidden_state = rearrange(hidden_state,"b (f n) c -> b c f n",f=f,n=h*w)


    indices = torch.arange(h * w).reshape(h, w)
    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    
    hidden_state = hidden_state[:, :, :, indices]
    hidden_state = rearrange(hidden_state,"b c f (h w) -> (b f) (h w) c",h=h,w=w)
    # (b f) (h w) c
    return hidden_state.contiguous()

def Continue_FWH(hidden_state):
    b, c, f, h, w = hidden_state.shape
    indices = torch.arange(w * h).reshape(w, h)
    

    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    
    hidden_state = rearrange(hidden_state, "b c f h w -> b c f (w h)")
    hidden_state = hidden_state[:, :, :, indices]
    

    frame_indices = torch.arange(f * w * h).reshape(f, w * h)
    frame_indices[1::2, :] = frame_indices[1::2, :].flip(-1)
    frame_indices = frame_indices.flatten()
    
    hidden_state = rearrange(hidden_state, "b c f n -> b (f n) c")
    hidden_state = hidden_state[:, frame_indices, :,]
    # b (f w h) c
    return hidden_state.contiguous()


def Restore_FWH(hidden_state, f, h, w):

    frame_indices = torch.arange(f * w * h).reshape(f, w * h)
    frame_indices[1::2, :] = frame_indices[1::2, :].flip(-1)
    frame_indices = frame_indices.flatten()
    hidden_state = hidden_state[:, frame_indices.argsort(), :]
    hidden_state = rearrange(hidden_state, "b (f n) c -> b c f n", f=f, n=w*h)
    

    indices = torch.arange(w * h).reshape(w, h)
    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    
    hidden_state = hidden_state[:, :, :, indices.argsort()]
    hidden_state = rearrange(hidden_state, "b c f (w h) -> (b f) (h w) c", w=w, h=h)
    # (b f) (h w) c
    return hidden_state.contiguous()





def Continue_HWF(hidden_state):
    b, c, f, h, w = hidden_state.shape
    indices = torch.arange(w * f).reshape(w, f)
    
    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    hidden_state = rearrange(hidden_state, "b c f h w -> b c h (w f)")
    hidden_state = hidden_state[:, :, :, indices]
  
    frame_indices = torch.arange(h * w * f).reshape(h, w * f)
    frame_indices[1::2, :] = frame_indices[1::2, :].flip(-1)
    frame_indices = frame_indices.flatten()
    
    hidden_state = rearrange(hidden_state, "b c h n -> b (h n) c")
    hidden_state = hidden_state[:, frame_indices, :,]
    # b (h,w,f) c
    return hidden_state.contiguous()


def Restore_HWF(hidden_state, f, h, w):
    b, seq, c = hidden_state.shape
 
    frame_indices = torch.arange(h * w * f).reshape(h, w * f)
    frame_indices[1::2, :] = frame_indices[1::2, :].flip(-1)
    frame_indices = frame_indices.flatten()
    hidden_state = hidden_state[:, frame_indices, :]
    hidden_state = rearrange(hidden_state, "b (h n) c -> b c h n", h=h, n=w*f)
  
    indices = torch.arange(w * f).reshape(w, f)
    indices[1::2, :] = indices[1::2, :].flip(-1)
    indices = indices.flatten()
    hidden_state = hidden_state[:, :, :, indices.argsort()]
    hidden_state = rearrange(hidden_state, "b c h (w f) -> (b f) (h w) c", w=w, f=f)
    # (b f) (h w) c
    return hidden_state.contiguous()



