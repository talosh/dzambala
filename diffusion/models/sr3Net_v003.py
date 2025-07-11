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

class FlownetDeep(nn.Module):
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

class GammaEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.linear = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()

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
                        beta_end = 4e-4,
                        image_dims = (3, 128, 128),
                        time_steps = 2048):
                
                super().__init__()
                self.time_steps = time_steps

                self.image_dims = image_dims
                c, h, w = self.image_dims
                self.img_size, self.input_channels = h, c

                self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
                self.alphas = 1 - self.betas
                self.alpha_hats = torch.cumprod(self.alphas, dim = -1)

                self.model = FlownetDeep(2*c, c = 96)

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