# **Network**
#
# Defines a 1D UNet architecture `ConditionalUnet1D`
# and a 2D UNet architecture `ConditionalUnet2D`
# as the noies prediction networks
#
# Components
# - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# - `Downsample1d` Strided convolution to reduce temporal resolution
# - `Upsample1d` Transposed convolution to increase temporal resolution
# - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.
# 1D UNet is adapted from Diffusion Policy - https://arxiv.org/abs/2303.04137
# 2D UNet is adapted from Thomas Capelle - https://github.com/tcapelle


from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x

# class SelfAttention(nn.Module):
#     def __init__(self, channels):
#         super(SelfAttention, self).__init__()
#         self.channels = channels        
#         self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x):
#         size = x.shape[-1]
#         x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
#         super().__init__()
#         self.residual = residual
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, mid_channels),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, out_channels),
#         )

#     def forward(self, x):
#         if self.residual:
#             return F.gelu(x + self.double_conv(x))
#         else:
#             return self.double_conv(x)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,
#                 out_channels
#             ),
#         )

#     def forward(self, x, t):
#         x = self.maxpool_conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.conv = nn.Sequential(
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels, in_channels // 2),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(
#                 emb_dim,
#                 out_channels
#             ),
#         )

#     def forward(self, x, skip_x, t):
#         x = self.up(x)
#         x = torch.cat([skip_x, x], dim=1)
#         x = self.conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb


# class UNet(nn.Module):
#     def __init__(self, input_dim = 32, c_in=3, c_out=3, remove_deep_conv=False):
#         super().__init__()
#         self.time_dim = 256
#         self.input_dim = input_dim
#         self.remove_deep_conv = remove_deep_conv
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256)


#         if remove_deep_conv:
#             self.bot1 = DoubleConv(256, 256)
#             self.bot3 = DoubleConv(256, 256)
#         else:
#             self.bot1 = DoubleConv(256, 512)
#             self.bot2 = DoubleConv(512, 512)
#             self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

#     def pos_encoding(self, t, channels):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def unet_forwad(self, x, t):
#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)

#         x4 = self.bot1(x4)
#         if not self.remove_deep_conv:
#             x4 = self.bot2(x4)
#         x4 = self.bot3(x4)

#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output
    
#     def forward(self, x, t):
#         t = t.unsqueeze(-1)
#         t = self.pos_encoding(t, self.time_dim)
#         return self.unet_forwad(x, t)


# class ConditionalUnet2D(UNet):
#     def __init__(self, input_dim = 64, c_in=3, c_out=3, global_cond_dim=None, **kwargs):
#         super().__init__(input_dim, c_in, c_out, **kwargs)
#         time_dim = self.time_dim
#         if global_cond_dim is not None:
#             self.label_emb = nn.Embedding(global_cond_dim, time_dim)

#     def forward(self, x, t, y=None):
#         t = t.unsqueeze(-1)
#         t = self.pos_encoding(t, self.time_dim)
#         if y is not None:
#             y_emb = self.label_emb(y)
#             t = t + y_emb
#         return self.unet_forwad(x, t)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, input_dim = 32, c_in=3, c_out=3, remove_deep_conv=False):
        super().__init__()
        self.time_dim = 128
        self.input_dim = input_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128)


        if remove_deep_conv:
            self.bot1 = DoubleConv(128, 128)
            self.bot3 = DoubleConv(128, 128)
        else:
            self.bot1 = DoubleConv(128, 256)
            self.bot2 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32)
        self.up3 = Up(64, 32)
        self.sa6 = SelfAttention(32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class ConditionalUnet2D(UNet):
    def __init__(self, input_dim = 64, c_in=3, c_out=3, global_cond_dim=None, **kwargs):
        super().__init__(input_dim, c_in, c_out, **kwargs)
        time_dim = self.time_dim
        if global_cond_dim is not None:
            self.label_emb = nn.Embedding(global_cond_dim, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            y_emb = self.label_emb(y)
            t = t + y_emb
        return self.unet_forwad(x, t)
    
# Just for unit tests
class TesterModel(nn.Module):
  def __init__(self, input_shape, c):
    super(TesterModel, self).__init__()
    x, y = input_shape
    self.fc1 = nn.Linear(x * y + c, x * y)
  
  def forward(self, input, t, c):
    input = input + (t.view(-1, 1, 1))/100
    x = input.flatten(1)
    x = torch.cat([x, c], dim=1)
    x = self.fc1(x)

    return x.reshape(input.shape)