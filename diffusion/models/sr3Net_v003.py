import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

class conv_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, embedding_dims = None):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_planes, 
            out_planes, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding, 
            dilation=dilation,
            padding_mode = 'zeros',
            bias=True
        )
        
        self.embedding_dims = embedding_dims if embedding_dims else out_planes
        
        # self.embedding = nn.Embedding(time_steps, embedding_dim = self.embedding_dims)
        self.embedding = GammaEncoding(self.embedding_dims)
        # switch to nn.Embedding if you want to pass in timestep instead; but note that it should be of dtype torch.long
        self.act = torch.nn.PReLU(out_planes, 0.2)

        
    def forward(self, inputs, time = None):
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # print(f"time embed shape => {time_embedding.shape}")
        x = self.conv1(inputs)
        x = self.act(x)
        x = x + time_embedding
        return x


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
        self.relu = torch.nn.PReLU(c, 0.2)
        self.embedding_dims = c
        self.embedding = GammaEncoding(self.embedding_dims)

    def forward(self, x):
        time = x[1]
        x = x[0]
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        x = self.relu(self.conv(x) * self.beta + x) + time_embedding
        return x, time

class UpMix(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep):
        
        return self.relu(self.conv(x_deep) * self.beta + x)

class Mix(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
        self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep):
        return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

class DownMix(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(cd, 0.2)

    def forward(self, x, x_deep):
        return self.relu(self.conv(x) * self.beta + x_deep)

class FlownetDeepOld(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)
        self.conv0 = conv_block(in_planes, c//2, 3, 2, 1)
        self.conv1 = conv_block(c//2, c, 3, 2, 1)
        self.conv2 = conv_block(c, cd, 3, 2, 1)
        self.convblock1 = torch.nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.convblock2 = torch.nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.convblock3 = torch.nn.Sequential(
            ResConv(c),
            ResConv(c),
        )
        self.convblock1f = torch.nn.Sequential(
            ResConv(c//2),
            ResConv(c//2),
            ResConv(c//2),
            ResConv(c//2),
        )
        self.convblock2f = torch.nn.Sequential(
            ResConv(c//2),
            ResConv(c//2),
            ResConv(c//2),
        )
        self.convblock3f = torch.nn.Sequential(
            ResConv(c//2),
            ResConv(c//2),
        )
        self.convblock_last = torch.nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.convblock_last_shallow = torch.nn.Sequential(
            ResConv(c//2),
            ResConv(c//2),
            ResConv(c//2),
            ResConv(c//2),
        )
        self.convblock_deep1 = torch.nn.Sequential(
            ResConv(cd),
            ResConv(cd),
            ResConv(cd),
            ResConv(cd),
        )
        self.convblock_deep2 = torch.nn.Sequential(
            ResConv(cd),
            ResConv(cd),
            ResConv(cd),
        )
        self.convblock_deep3 = torch.nn.Sequential(
            ResConv(cd),
            ResConv(cd),
        )
        
        self.mix1 = UpMix(c, cd)
        self.mix1f = DownMix(c//2, c)
        self.mix2 = UpMix(c, cd)
        self.mix2f = DownMix(c//2, c)
        self.mix3 = Mix(c, cd)
        self.mix3f = DownMix(c//2, c)
        self.mix4f = DownMix(c//2, c)
        self.revmix1 = DownMix(c, cd)
        self.revmix1f = UpMix(c//2, c)
        self.revmix2 = DownMix(c, cd)
        self.revmix2f = UpMix(c//2, c)
        self.revmix3f = UpMix(c//2, c)
        self.mix4 = Mix(c//2, c)
        self.lastconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c//2, 3, 4, 2, 1),
        )
        self.maxdepth = 8

    def forward(self, x, time, scale=1):
        n, c, h, w = x.shape
        sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

        ph = self.maxdepth - (sh % self.maxdepth)
        pw = self.maxdepth - (sw % self.maxdepth)
        padding = (0, pw, 0, ph)

        # x = torch.cat((img0, f0), 1)
        x = torch.nn.functional.pad(x, padding)

        feat = self.conv0(x, time)
        featF, _ = self.convblock1f((feat, time))

        feat = self.conv1(feat, time)
        feat_deep = self.conv2(feat, time)

        feat, _ = self.convblock1((feat, time))
        feat_deep, _ = self.convblock_deep1((feat_deep, time))
        
        feat = self.mix1f(featF, feat)
        feat_tmp = self.mix1(feat, feat_deep)
        feat_deep = self.revmix1(feat, feat_deep)

        featF = self.revmix1f(featF, feat_tmp)

        featF, _ = self.convblock2f((featF, time))
        feat, _ = self.convblock2((feat_tmp, time))
        feat_deep, _ = self.convblock_deep2((feat_deep, time))

        feat = self.mix2f(featF, feat)
        feat_tmp = self.mix2(feat, feat_deep)
        feat_deep = self.revmix2(feat, feat_deep)
        featF = self.revmix2f(featF, feat_tmp)

        featF, _ = self.convblock3f((featF, time))
        feat, _ = self.convblock3((feat_tmp, time))
        feat_deep, _ = self.convblock_deep3((feat_deep, time))
        feat = self.mix3f(featF, feat)
        feat = self.mix3(feat, feat_deep)
        
        featF = self.revmix3f(featF, feat)

        feat, _ = self.convblock_last((feat, time))
        featF, _ = self.convblock_last_shallow((featF, time))

        feat = self.mix4(featF, feat)

        feat = self.lastconv(feat)
        # feat = torch.nn.functional.interpolate(feat[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
        return feat[:, :, :sh, :sw]

class myPReLU(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 0.2), requires_grad=True)
        self.beta = self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 0.69), requires_grad=True)
        self.prelu = torch.nn.PReLU(c, 0.2)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        alpha = self.alpha.clamp(min=1e-8)
        x = x / alpha - self.beta
        tanh_x = self.tanh(x)
        x = torch.where(
            x > 0, 
            x, 
            tanh_x + abs(tanh_x) * self.prelu(x)
        )
        return alpha * (x + self.beta)

class HighPassFilter(torch.nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        self.register_buffer('gkernel', self.gauss_kernel())

    def gauss_kernel(self, channels=1):
        kernel = torch.tensor([
            [1., 4., 6., 4., 1],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def rgb_to_luminance(self, rgb):
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
        return (rgb * weights).sum(dim=1, keepdim=True)

    def normalize(self, tensor, min_val, max_val):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        tensor = tensor * (max_val - min_val) + min_val
        return tensor

    def forward(self, img):
        img = self.rgb_to_luminance(img)
        hp = img - self.conv_gauss(img, self.gkernel) + 0.5
        hp = torch.clamp(hp, 0.48, 0.52)
        hp = self.normalize(hp, 0, 1)
        return hp

class HighPassFilter3(torch.nn.Module):
    def __init__(self):
        super(HighPassFilter3, self).__init__()
        self.register_buffer('gkernel', self.gauss_kernel())

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([
            [1., 4., 6., 4., 1],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def normalize(self, tensor, min_val, max_val):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        tensor = tensor * (max_val - min_val) + min_val
        return tensor

    def forward(self, img):
        hp = img - self.conv_gauss(img, self.gkernel) + 0.5
        hp = torch.clamp(hp, 0.48, 0.52)
        hp = self.normalize(hp, 0, 1)
        return hp
class FeatureModulator(torch.nn.Module):
    def __init__(self, scalar_dim, feature_channels):
        super().__init__()
        self.scale_net = torch.nn.Sequential(
            torch.nn.Linear(scalar_dim, feature_channels),
            # torch.nn.PReLU(feature_channels, 1),  # or no activation
        )
        self.shift_net = torch.nn.Linear(scalar_dim, feature_channels)
        self.c = feature_channels

    def forward(self, x_scalar, features):
        scale = self.scale_net(x_scalar).view(-1, self.c, 1, 1)
        shift = self.shift_net(x_scalar).view(-1, self.c, 1, 1)
        return features * scale + shift

class FourierChannelAttention(torch.nn.Module):
    def __init__(self, c, latent_dim, out_channels, bands = 11, norm = False):
        super().__init__()

        self.bands = bands
        self.norm = norm
        self.c = c

        self.alpha = torch.nn.Parameter(torch.full((1, c, 1, 1), 1.0), requires_grad=True)

        self.precomp = torch.nn.Sequential(
            torch.nn.Conv2d(c + 2, c, 3, 1, 1),
            torch.nn.PReLU(c, 0.2),
            torch.nn.Conv2d(c, c, 3, 1, 1),
            torch.nn.PReLU(c, 0.2),
        )

        self.encoder = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d((bands, bands)),
            torch.nn.Conv2d(c, out_channels, 1, 1, 0),
            torch.nn.PReLU(out_channels, 0.2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(bands * bands * out_channels, latent_dim),
            torch.nn.PReLU(latent_dim, 0.2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, bands * bands * c),
            torch.nn.Sigmoid(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, c),
            torch.nn.Sigmoid(),
        )

    def normalize_fft_magnitude(self, mag, sh, sw, target_size=(64, 64)):
        """
        mag: [B, C, sh, sw]
        Returns: [B, C, Fy, Fx]
        """
        B, C, _, _ = mag.shape
        Fy, Fx = target_size

        mag_reshaped = mag.view(B * C, 1, sh, sw)
        norm_mag = torch.nn.functional.interpolate(
            mag_reshaped, size=(Fy, Fx), mode='bilinear', align_corners=False
        )
        norm_mag = norm_mag.view(B, C, Fy, Fx)
        return norm_mag

    def denormalize_fft_magnitude(self, norm_mag, sh, sw):
        """
        norm_mag: [B, C, Fy, Fx]
        Returns: [B, C, sh, sw]
        """
        B, C, Fy, Fx = norm_mag.shape

        norm_mag = norm_mag.view(B * C, 1, Fy, Fx)
        mag = torch.nn.functional.interpolate(
            norm_mag, size=(sh, sw), mode='bilinear', align_corners=False
        )
        mag = mag.view(B, C, sh, sw)
        return mag
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2 + 1]
        _, _, sh, sw = x_fft.shape

        mag = x_fft.abs()
        phase = x_fft.angle()

        if self.norm:
            mag_n = self.normalize_fft_magnitude(mag, sh, sw, target_size=(64, 64))
        else:
            mag_n = torch.nn.functional.interpolate(
                mag, 
                size=(64, 64), 
                mode="bilinear",
                align_corners=False, 
                )

        mag_n = torch.log1p(mag_n) + self.alpha * mag_n
        grid_x = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 1, 64).expand(B, 1, 64, 64)
        grid_y = torch.linspace(0, 1, 64, device=x.device).view(1, 1, 64, 1).expand(B, 1, 64, 64)
        mag_n = self.precomp(torch.cat([mag_n, grid_x, grid_y], dim=1))

        latent = self.encoder(mag_n)

        spat_at = self.fc1(latent).view(-1, self.c, self.bands, self.bands)
        spat_at = spat_at / 0.4 + 0.5
        if self.norm:
            spat_at = self.denormalize_fft_magnitude(spat_at, sh, sw)
        else:
            spat_at = torch.nn.functional.interpolate(
                spat_at, 
                size=(sh, sw), 
                mode="bilinear",
                align_corners=False, 
                )

            mag = mag * spat_at.clamp(min=1e-6)
            x_fft = torch.polar(mag, phase)
            x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')

        chan_scale = self.fc2(latent).view(-1, self.c, 1, 1) + 0.1
        x = x * chan_scale.clamp(min=1e-6)
        return x

class HeadAtt(torch.nn.Module):
    def __init__(self, c=48):
        super(HeadAtt, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(4, c, 5, 2, 2, padding_mode = 'reflect'),
            torch.nn.ELU(),
            torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
            torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
        )
        self.attn = FourierChannelAttention(c, c, 24, norm=False)
        self.lastconv = torch.nn.ConvTranspose2d(c, 8, 4, 2, 1)
        self.hpass = HighPassFilter()
        self.maxdepth = 2

    def forward(self, x):
        hp = self.hpass(x)
        x = torch.cat((x, hp), 1)

        n, c, h, w = x.shape
        ph = self.maxdepth - (h % self.maxdepth)
        pw = self.maxdepth - (w % self.maxdepth)
        padding = (0, pw, 0, ph)
        x = torch.nn.functional.pad(x, padding)

        x = self.encode(x)
        x = self.attn(x)
        x = self.lastconv(x)
        return x[:, :, :h, :w]

class ResConvEmb(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)
        self.embedding_dims = c
        self.embedding = GammaEncoding(self.embedding_dims)
        self.mlp = FeatureModulator(c, c)

    def forward(self, x):
        time = x[1]
        x = x[0]
        time_embedding = self.embedding(time) # .view(-1, self.embedding_dims, 1, 1)
        x = self.relu(self.mlp(time_embedding, self.conv(x)) * self.beta + x)
        return x, time

class ResConvEmb_(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
        self.relu = torch.nn.PReLU(c, 0.2)
        self.embedding_dims = c
        self.embedding = GammaEncoding(self.embedding_dims)

    def forward(self, x):
        time = x[1]
        x = x[0]
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        x = self.relu(self.conv(x) * self.beta + x) + time_embedding
        return x, time

class ResConvAtt(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)
        self.embedding_dims = c
        self.embedding = GammaEncoding(self.embedding_dims)
        self.mlp = FeatureModulator(c, c)
        self.attn = FourierChannelAttention(c, c, 24, norm=False)

    def forward(self, x):
        time = x[1]
        x = self.attn(x[0])
        time_embedding = self.embedding(time) # .view(-1, self.embedding_dims, 1, 1)
        x = self.relu(self.mlp(time_embedding, self.conv(x)) * self.beta + x)
        return x, time

class UpMix(torch.nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep, t):
        return self.relu(self.conv(x_deep) * self.beta + x)

class Mix(torch.nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
        self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep, t):
        return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

class MixSame(torch.nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(cd, c, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)

    def forward(self, x, x_deep, t):
        return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

class DownMix(torch.nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(cd, 0.2)

    def forward(self, x, x_deep, t):
        return self.relu(self.conv(x) * self.beta + x_deep)

class FlownetDeep(torch.nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, c//2, 5, 2, 2),
            torch.nn.PReLU(c//2),
            )                
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(c//2, c, 5, 2, 2, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
            )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(cd, 0.2),     
        )

        self.conv_hp = torch.nn.Sequential(
            torch.nn.Conv2d(3, c//8, 3, 1, 1),
            torch.nn.PReLU(c//8, 0.2),
        )

        self.convblock_hp = torch.nn.Sequential(
            ResConvEmb(c//8),
            ResConvEmb(c//8),
        )

        self.convblock1 = torch.nn.Sequential(
            ResConvAtt(c),
            ResConvEmb(c),
            ResConvEmb(c),
        )

        self.convblock2 = torch.nn.Sequential(
            ResConvEmb(c),
            ResConvEmb(c),
        )
        self.convblock3 = torch.nn.Sequential(
            ResConvEmb(c),
        )
        self.convblock1f = torch.nn.Sequential(
            ResConvEmb(c//2),
            ResConvEmb(c//2),
            ResConvEmb(c//2),
        )
        self.convblock2f = torch.nn.Sequential(
            ResConvEmb(c//2),
            ResConvEmb(c//2),
        )
        self.convblock3f = torch.nn.Sequential(
            ResConvEmb(c//2),
            ResConvEmb(c//2),
        )
        self.convblock_last = torch.nn.Sequential(
            ResConvEmb(c),
            ResConvEmb(c),
            ResConvEmb(c),
            ResConvEmb(c),
        )
        self.convblock_last_shallow = torch.nn.Sequential(
            ResConvEmb(c//2),
            ResConvEmb(c//2),
            ResConvEmb(c//2),
            ResConvEmb(c//2),
        )
        self.convblock_deep1 = torch.nn.Sequential(
            ResConvAtt(cd),
            ResConvEmb(cd),
            ResConvEmb(cd),
            ResConvEmb(cd),
        )
        self.convblock_deep2 = torch.nn.Sequential(
            ResConvAtt(cd),
            ResConvEmb(cd),
            ResConvEmb(cd),
        )
        self.convblock_deep3 = torch.nn.Sequential(
            ResConvEmb(cd),
            ResConvEmb(cd),
        )

        # self.hpass = HighPassFilter3()

        self.mix1 = UpMix(c, cd)
        self.mix1f = DownMix(c//2, c)
        self.mix2 = UpMix(c, cd)
        self.mix2f = DownMix(c//2, c)
        self.mix3 = Mix(c, cd)
        self.mix3f = DownMix(c//2, c)
        self.mix4f = DownMix(c//2, c)
        self.revmix1 = DownMix(c, cd)
        self.revmix1f = UpMix(c//2, c)
        self.revmix2 = DownMix(c, cd)
        self.revmix2f = UpMix(c//2, c)
        self.revmix3f = UpMix(c//2, c)
        self.mix4 = Mix(c//2, c)
        self.mix_hp = MixSame(c//2, c//8)

        self.up = torch.nn.ConvTranspose2d(c//2, c//2, 4, 2, 1) # torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.lastconv = torch.nn.Conv2d(c//2, 3, 5, 1, 2)
        self.maxdepth = 8

        # self.register_buffer("forward_counter1", torch.tensor(0, dtype=torch.long))
        # self.mix_ratio = 0.


    def resize_min_side(self, tensor, size):
        B, C, H, W = tensor.shape

        if H <= W:
            new_h = size
            new_w = int(round(W * (56 / H)))
        else:
            new_w = size
            new_h = int(round(H * (56 / W)))

        return torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=True)

    def forward(self, x, ratio):
        n, c, h, w = x.shape
        ph = self.maxdepth - (h % self.maxdepth)
        pw = self.maxdepth - (w % self.maxdepth)
        padding = (0, pw, 0, ph)

        x = torch.nn.functional.pad(x, padding)

        ratio = ratio.to(x.device)

        feat = self.conv0(x)
        x = self.conv_hp(x[:, :3, :, :])
        x, _ = self.convblock_hp((x, ratio))

        featF, _ = self.convblock1f((feat, ratio))

        feat = self.conv1(feat)
        feat_deep = self.conv2(feat)

        feat, _ = self.convblock1((feat, ratio))
        feat_deep, _ = self.convblock_deep1((feat_deep, ratio))

        feat = self.mix1f(featF, feat, ratio)
        feat_tmp = self.mix1(feat, feat_deep, ratio)
        feat_deep = self.revmix1(feat, feat_deep, ratio)
        featF = self.revmix1f(featF, feat_tmp, ratio)

        featF, _ = self.convblock2f((featF, ratio))
        feat, _ = self.convblock2((feat_tmp, ratio))
        feat_deep, _ = self.convblock_deep2((feat_deep, ratio))

        feat = self.mix2f(featF, feat, ratio)
        feat_tmp = self.mix2(
            feat,
            feat_deep,
            ratio
            )
        feat_deep = self.revmix2(
            feat,
            feat_deep,
            ratio
            )
        featF = self.revmix2f(featF, feat_tmp, ratio)

        featF, _ = self.convblock3f((featF, ratio))
        feat, _ = self.convblock3((feat_tmp, ratio))
        feat_deep, _ = self.convblock_deep3((feat_deep, ratio))
        
        feat = self.mix3f(featF, feat, ratio)
        feat = self.mix3(
            feat,
            feat_deep,
            ratio
            )
        
        featF = self.revmix3f(featF, feat, ratio)

        feat, _ = self.convblock_last((feat, ratio))
        featF, _ = self.convblock_last_shallow((featF, ratio))

        feat = self.mix4(featF, feat, ratio)
        feat = self.up(feat)
        feat = self.mix_hp(feat, x, ratio)

        feat = self.lastconv(feat)
        feat = ( feat + 1 ) / 2
        return feat[:, :, :h, :w]

class GammaEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.act = torch.nn.PReLU(dim) # nn.LeakyReLU()

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return self.act(self.linear(encoding))

class Model:
    info = {
        'name': 'sr3Net_v003',
        'file': 'sr3Net_v003.py',
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module

        class DiffusionModel(nn.Module):
            def __init__(self, 
                        beta_start = 1e-5, 
                        beta_end = 4e-2,
                        image_dims = (3, 128, 128),
                        time_steps = 1024):
                
                super().__init__()
                self.time_steps = time_steps

                self.image_dims = image_dims
                c, h, w = self.image_dims
                self.img_size, self.input_channels = h, c

                self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
                self.alphas = 1 - self.betas
                self.alpha_hats = torch.cumprod(self.alphas, dim = -1)

                self.model = FlownetDeep(2*c + 8, c = 96)
                self.encode = HeadAtt()

            def add_noise(self, x, ts):
                # 'x' and 'ts' are expected to be batched

                noise = torch.randn_like(x)
                # print(x.shape, noise.shape)
                noised_examples = []
                for i, t in enumerate(ts):
                    alpha_hat_t = self.alpha_hats[t]
                    noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])

                return torch.stack(noised_examples), noise

            def forward(self, x, t):
                n, c, h, w = x.shape
                f = torch.zeros(n, 8, h, w, device=x.device, dtype=x.dtype)
                f[:, :, :, :w//2] = self.encode(x[:, :3, :, :w//2])
                x = torch.cat([f, x], dim = 1)
                return self.model(x, t)

        self.model = DiffusionModel
        self.training_model = DiffusionModel

    @staticmethod
    def get_info():
        return Model.info

    @staticmethod
    def get_name():
        return Model.info.get('name')

    def get_model(self):
        return self.model

    def get_training_model(self):
        return self.training_model