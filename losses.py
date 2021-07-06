import torch
from torch import Tensor
import torch.nn.functional as F


def d_loss(d_real: Tensor, d_fake: Tensor, d_gray: Tensor, d_smooth: Tensor) -> Tensor:
    err_real = torch.mean((d_real - 1)**2)
        
    err_fake = torch.mean(d_fake**2)

    err_gray = torch.mean(d_gray**2)

    err_smooth = torch.mean(d_smooth**2)

    return 1.7*err_d_real + 1.7*err_d_fake + 1.7*gray_loss + 1.0*err_d_smooth

def g_loss(d_fake: Tensor) -> Tensor:
    return torch.mean((d_fake - 1)**2)

def con_loss(input: Tensor, fake: Tensor) -> Tensor:
    return torch.mean(torch.abs(input - fake))

def gram(input: Tensor) -> Tensor:
    N, C, H, W = input.size()
    input = input.reshape(N*C, H*W)
    G = torch.mm(input, input.t())
    return G.div(N*C*H*W)

def style_loss(input: Tensor, fake: Tensor) -> Tensor:
    return torch.mean(torch.abs(gram(input) - gram(fake)))

def rgb2yuv(input: Tensor) -> Tensor:
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b

    out = torch.stack([y, u, v], -3)
    return out

def color_loss(input: Tensor, fake: Tensor) -> Tensor:
    input = rgb2yuv(input)
    fake = rgb2yuv(fake)

    return torch.mean(torch.abs(input[:,:,:,0] - fake[:,:,:,0])) + F.smooth_l1_loss(input[:,:,:,1], fake[:,:,:,1]) + F.smooth_l1_loss(input[:,:,:,2], fake[:,:,:,2])