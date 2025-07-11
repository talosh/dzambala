import torch

class ACESCCTtoACESCG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("const_cond1", torch.tensor(0.155251141552511))
        self.register_buffer("const_cond2", (torch.log2(torch.tensor(65504.0)) + 9.72) / 17.52)
        self.register_buffer("const_cond3", torch.tensor(65504.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        condition = image < self.const_cond1
        value_if_true = (image - 0.0729055341958155) / 10.5402377416545
        ACEScg = torch.where(condition, value_if_true, image)

        condition = (image >= self.const_cond1) & (image < self.const_cond2)
        value_if_true = torch.exp2(image * 17.52 - 9.72)
        ACEScg = torch.where(condition, value_if_true, ACEScg)

        ACEScg = torch.clamp(ACEScg, max=self.const_cond3)

        return ACEScg

class ACESCGtoACESCCT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("const_cond1", torch.tensor(0.0078125))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        condition = image <= self.const_cond1
        value_if_true = image * 10.5402377416545 + 0.0729055341958155 
        ACEScct = torch.where(condition, value_if_true, image)
        
        condition = image > self.const_cond1
        value_if_true = (torch.log2(image) + 9.72) / 17.52
        ACEScct = torch.where(condition, value_if_true, ACEScct)

        return ACEScct

class CompressHighlights(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        condition = image > 1
        value_if_true = (torch.log2(image) / 4) + 1
        compressed = torch.where(condition, value_if_true, image)
        return compressed

class ExpandHighlights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.register_buffer("const_cond1", torch.tensor(1))
        # self.register_buffer("const_cond2", (torch.log2(torch.tensor(65504.0)) / 8) + 1)
        # self.register_buffer("const_cond3", torch.tensor(65504.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        condition = image > 1
        value_if_true = torch.exp2((image - 1) * 4)
        expanded = torch.where(condition, value_if_true, image)
        expanded = torch.clamp(expanded, max=65504.0)
        return expanded

class ACEScgToOKLab(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define AP1 (ACEScg) to LMS transformation matrix
        self.register_buffer("M1", torch.tensor([
            [1.704858676289160, -0.130076824208823, -0.023964072927574],
            [-0.621716021885330, 1.140735774822510, -0.128975508299318],
            [-0.083299371729057, -0.010559801677511, 1.153014018916860]
        ], dtype=torch.float32))

        # Define LMS to OKLab transformation matrix
        self.register_buffer("M2", torch.tensor([
            [0.4122214708, 0.2119034982, 0.0883024619],
            [0.5363325363, 0.6806995451, 0.2817188376],
            [0.0514459929, 0.1073969566, 0.6299787005]
        ], dtype=torch.float32))

        # Define OKLab conversion coefficients
        self.register_buffer("M3", torch.tensor([
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850,-2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099,-0.8086757660]
        ], dtype=torch.float32))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape

        img = img.permute(0, 2, 3, 1).reshape(-1, 3)
        img = torch.matmul(img, self.M1)
        img = torch.matmul(img, self.M2)
        img = img.sign() * img.abs().pow(1 / 3)
        img = torch.matmul(img, self.M3)

        return img.reshape(B, H, W, 3).permute(0, 3, 1, 2)

class OKLabToACEScg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("M1", torch.tensor([
            [1.0000000000, 1.0000000000, 1.0000000000],
            [0.3963377774,-0.1055613458,-0.0894841775],
            [0.2158037573,-0.0638541728,-1.2914855480]
        ], dtype=torch.float32))
        self.register_buffer("M2", torch.tensor([
            [ 4.0767416621,-1.2684380046, 0.0041960863],
            [-3.3077115913, 2.6097574011,-0.7034186147],
            [ 0.2309699292,-0.3413193965, 1.7076147010]
        ], dtype=torch.float32))
        self.register_buffer("M3", torch.tensor([
            [0.613130111351449, 0.0701050973342151, 0.0205851228833001],
            [0.339523761133735, 0.916356903394254, 0.109559786350199],
            [0.0474050234857193, 0.0134571284251877, 0.869782709459968]
        ], dtype=torch.float32))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape

        img = img.permute(0, 2, 3, 1).reshape(-1, 3)
        img = torch.matmul(img, self.M1)
        img = img.sign() * img.abs().pow(3)
        img = torch.matmul(img, self.M2)
        img = torch.matmul(img, self.M3)

        return img.reshape(B, H, W, 3).permute(0, 3, 1, 2)

class ApplyLUT(torch.nn.Module):
    """
    Apply a 3D LUT to an image using trilinear interpolation.
    """
    def __init__(self, size=33):
        """
        Initializes the LUT module without a predefined LUT.
        The LUT should be assigned dynamically before calling forward().
        """
        super().__init__()

        values = torch.linspace(0, 1, steps=size)
        r, g, b = torch.meshgrid(values, values, values, indexing='ij')  # 3D grid
        # Stack channels to form the LUT (size, size, size, 3)
        lut = torch.stack([r, g, b], dim=-1)  # (size, size, size, 3)
        lut = lut.cuda()
        self.register_buffer("id_lut", lut.clone().detach())
        self.register_buffer("lut", lut)

    def set_lut(self, lut: torch.Tensor, ratio: float = 1.0):
        """
        Dynamically assigns a LUT.

        Args:
            lut (torch.Tensor): LUT tensor of shape (L, L, L, 3)
        """
        self.lut = lut * ratio + self.id_lut * (1 - ratio)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the LUT to an image.

        Args:
            image (torch.Tensor): Input tensor of shape (B, 3, H, W), RGB values in range [0, 1]
        
        Returns:
            torch.Tensor: LUT-applied output tensor of shape (B, 3, H, W)
        """
        # if self.lut is None:
        #    raise ValueError("LUT is not set. Use `.set_lut(lut_tensor)` before calling forward.")

        image = image.permute(0, 2, 3, 1)  # Convert (B, 3, H, W) → (B, H, W, 3)
        B, H, W, C = image.shape
        assert C == 3, "Input image must have 3 channels (RGB)"

        L = self.lut.shape[0]  # LUT resolution

        # Scale image values to LUT index space
        image_scaled = image * (L - 1)

        # Get integer and fractional parts for interpolation
        idx = image_scaled.floor().long()
        frac = image_scaled - idx.float()

        # Clamp indices to be within LUT bounds
        idx0 = idx.clamp(0, L - 2)
        idx1 = (idx0 + 1).clamp(0, L - 1)

        # Gather LUT values at 8 surrounding corners (JIT-Compatible)
        c000 = self.lut[idx0[..., 0], idx0[..., 1], idx0[..., 2]]
        c100 = self.lut[idx1[..., 0], idx0[..., 1], idx0[..., 2]]
        c010 = self.lut[idx0[..., 0], idx1[..., 1], idx0[..., 2]]
        c110 = self.lut[idx1[..., 0], idx1[..., 1], idx0[..., 2]]
        c001 = self.lut[idx0[..., 0], idx0[..., 1], idx1[..., 2]]
        c101 = self.lut[idx1[..., 0], idx0[..., 1], idx1[..., 2]]
        c011 = self.lut[idx0[..., 0], idx1[..., 1], idx1[..., 2]]
        c111 = self.lut[idx1[..., 0], idx1[..., 1], idx1[..., 2]]

        # Expand `frac` for proper broadcasting across channels
        frac = frac.unsqueeze(-1)

        # Perform trilinear interpolation
        c00 = c000 * (1 - frac[..., 0, :]) + c100 * frac[..., 0, :]
        c01 = c001 * (1 - frac[..., 0, :]) + c101 * frac[..., 0, :]
        c10 = c010 * (1 - frac[..., 0, :]) + c110 * frac[..., 0, :]
        c11 = c011 * (1 - frac[..., 0, :]) + c111 * frac[..., 0, :]

        c0 = c00 * (1 - frac[..., 1, :]) + c10 * frac[..., 1, :]
        c1 = c01 * (1 - frac[..., 1, :]) + c11 * frac[..., 1, :]

        output = c0 * (1 - frac[..., 2, :]) + c1 * frac[..., 2, :]

        return output.permute(0, 3, 1, 2)  # Convert back to (B, 3, H, W)

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
            myPReLU(c),
            torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
            torch.nn.Conv2d(c, c, 3, 1, 1, padding_mode = 'reflect'),
            torch.nn.PReLU(c, 0.2),
        )
        self.attn = FourierChannelAttention(c, c, 24, norm=True)
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
        x = self.lastconv(x)[:, :, :h, :w]
        return x

class ResConvEmb(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)
        self.mlp = FeatureModulator(1, c)

    def forward(self, x):
        x_scalar = x[1]
        x = x[0]
        x = self.relu(self.mlp(x_scalar, self.conv(x)) * self.beta + x)
        return x, x_scalar

class ResConvAtt(torch.nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
        self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = torch.nn.PReLU(c, 0.2)
        self.mlp = FeatureModulator(1, c)
        self.attn = FourierChannelAttention(c, c, 24, norm=True)

    def forward(self, x):
        x_scalar = x[1]
        x = self.attn(x[0])
        x = self.relu(self.mlp(x_scalar, self.conv(x)) * self.beta + x)
        return x, x_scalar

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
            myPReLU(c//2),
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
            ResConvAtt(c//8),
            ResConvEmb(c//8),
        )

        self.convblock1 = torch.nn.Sequential(
            ResConvAtt(c),
            ResConvEmb(c),
            ResConvEmb(c),
            ResConvEmb(c),
        )

        self.convblock2 = torch.nn.Sequential(
            ResConvAtt(c),
            ResConvEmb(c),
            ResConvEmb(c),
        )
        self.convblock3 = torch.nn.Sequential(
            ResConvEmb(c),
            ResConvEmb(c),
        )
        self.convblock1f = torch.nn.Sequential(
            ResConvAtt(c//2),
            ResConvEmb(c//2),
            ResConvEmb(c//2),
            ResConvEmb(c//2),
        )
        self.convblock2f = torch.nn.Sequential(
            ResConvAtt(c//2),
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

        self.up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.lastconv = torch.nn.Sequential(
            torch.nn.Conv2d(c//2, 3, 5, 1, 2),
            torch.nn.Tanh()
        )
        self.maxdepth = 16

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

    def forward(self, img0, f0, ratio):
        n, c, h, w = img0.shape
        ph = self.maxdepth - (h % self.maxdepth)
        pw = self.maxdepth - (w % self.maxdepth)
        padding = (0, pw, 0, ph)

        x = torch.cat((img0, f0), 1)
        ratio = ratio.to(img0.device)

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

class Model:

    info = {
        'name': 'fNet_v001',
        'file': 'fNet_v001.py',
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        class focusNet(torch.nn.Module):
            def __init__(self, in_planes=3, c=12):
                super().__init__()
                self.encode = HeadAtt()
                self.focus = FlownetDeep(11, c=96)

            def forward(
                    self, 
                        x: torch.Tensor, 
                        ratio: torch.Tensor,
            ) -> torch.Tensor:
                x = torch.clamp(x[:, :3, :, :], 0, 1)
                f0 = self.encode(x)
                x = self.focus(x, f0, ratio)
                return x

        self.model = focusNet
        self.training_model = focusNet

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