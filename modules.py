import torch
from torch import nn
import torch.nn.functional as F


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1, bias=False):    
        super(ConvNormLReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(in_ch*expansion_ratio)
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out

    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0,1,0,1)),
            ConvNormLReLU(64, 64)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0,1,0,1)),            
            ConvNormLReLU(128, 128)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )    
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
        
    def forward(self, input):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)
        
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, nbchannels, nblayers):
        super().__init__()
        layers = []
        nbchannels //= 2
        layers.append(nn.Conv2d(3, nbchannels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(nblayers):
            layers.append(nn.Conv2d(nbchannels if i == 0 else nbchannels*2, nbchannels*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.LeakyReLU(0.2))
            
            layers.append(nn.Conv2d(nbchannels*2, nbchannels*4, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.GroupNorm(num_groups=1, num_channels=nbchannels*4, affine=True))
            layers.append(nn.LeakyReLU(0.2))

            nbchannels *= 2
        
        layers.append(nn.Conv2d(nbchannels*2, nbchannels*4, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=nbchannels*4, affine=True))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(nbchannels*4, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers = nn.Sequential(*layers)
        return

    def forward(self, x):
        return self.layers(x)
