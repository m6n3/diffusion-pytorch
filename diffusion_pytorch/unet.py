import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        # A feed-forward neural network is used for time-embeding instead of
        # sinusoidal embedding (used in the original paper).
        self.layers = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t):
        # t: [batch_size, 1]

        device = next(self.parameters())

        tembed = self.layers(t.float().to(device))
        # tembed: [batch_size, embed_dim]

        return tembed


# TODO: find a better name.
class Pre(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class ResNet(nn.Module):
    DROPOUT = 0.1

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.forward_path = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_ch),
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.time_path = nn.Sequential(nn.GELU(), nn.Linear(time_dim, out_ch))

        self.back_path = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
            nn.Dropout(p=ResNet.DROPOUT),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        # Used only to adjust the channel of the residual.
        self.res_path = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, tembed=None):
        # x: [batch_size, in_ch, H, W]
        # tembed: [batch_size, time_dim]
        h = self.forward_path(x)
        # h: [batch_size, out_ch, H, W]

        tembed = self.time_path(tembed).unsqueeze(-1).unsqueeze(-1)
        # tembed: [batch_size, out_ch, 1, 1]

        h += tembed
        # h: [batch_size, out_ch, H, W]

        h = self.back_path(h)
        # h: [batch_size, out_ch, H, W]

        res = self.res_path(x)
        # res: [batch_size, out_ch, H, W]

        return h + res


class Attn(nn.Module):
    """Attention module"""

    def __init__(self, in_ch):
        super().__init__()
        # Unlike the original paper (which uses a in_ch*in_ch linear transform),
        # we use Conv2d with a kernel size of 1 (a diagonal in_ch*in_ch linear transform).
        self.Q = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.K = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.V = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(in_ch, in_ch)

    def forward(self, x):
        # x: [batch_size, in_ch, H, W]

        B, C, H, W = x.shape
        query, key, value = (
            self.Q(x).view(B, -1, H * W),
            self.K(x).view(B, -1, H * W),
            self.V(x).view(B, -1, H * W),
        )
        # query,key,value: [batch_size, in_ch, H*W]

        query = query.transpose(1, 2)
        # query: [batch_size, H*W, in_ch]

        energy = torch.bmm(query, key) / int(C) ** (-0.5)
        # energy: [batch_size, H*W, H*W]

        attention = self.softmax(energy)  # softmax dim=-1
        # attention: [batch_size, H*W, H*W]

        out = torch.bmm(value, attention.transpose(1, 2))
        # out: [batch_size, in_ch, H*W]

        out = self.linear(out.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        # out: [batch_size, in_ch, H, W]

        return out + x


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # [B, C, H, W] -> [B, C, H/2, W/2]
        self.down_sample = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        # x: [batch_size, in_ch, H, W]

        out = self.down_sample(x)
        # out: [batch_size, in_ch, H/2, W/2]

        return out


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # [B, C, H, W] -> [B, C, 2*H, 2*W]
        self.up_sample = nn.ConvTranspose2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        # x: [batch_size, in_ch, H, W]

        out = self.up_sample(x)
        # out: [batch_size, in_ch, 2*H, 2*W]

        return out


class Down(nn.Module):
    def __init__(
        self, in_ch, out_ch, time_dim, heights_for_attn=(), add_downsampling=True
    ):
        super().__init__()
        # Apply attention only if input image's height (H) in listed in
        # `heights_for_attn`.
        self.heights_for_attn = heights_for_attn
        self.resnets = nn.ModuleList(
            [
                ResNet(in_ch=in_ch, out_ch=out_ch, time_dim=time_dim),
                ResNet(in_ch=out_ch, out_ch=out_ch, time_dim=time_dim),
            ]
        )
        self.attns = nn.ModuleList([Attn(in_ch=out_ch), Attn(in_ch=out_ch)])
        self.downsample = DownSample(out_ch) if add_downsampling else None

    def forward(self, x, temb):
        # x: [batch_size, in_ch, H, W]
        # temb: [batch_size, time_dim]

        h = x
        for resnet, attn in zip(self.resnets, self.attns):
            h = resnet(h, temb)
            # h: [batch_size, out_ch, H, W]

            if x.shape[2] in self.heights_for_attn:
                h = attn(h)
                # h: [batch_size, out-ch, H, W]

        if self.downsample:
            h = self.downsample(h)
            # h: [batch_size, out_ch, H/2, W/2]

        return h


class Middle(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.resnet1 = ResNet(in_ch=in_ch, out_ch=out_ch, time_dim=time_dim)
        self.attn = Attn(in_ch=out_ch)
        self.resnet2 = ResNet(in_ch=out_ch, out_ch=out_ch, time_dim=time_dim)

    def forward(self, x, temb):
        # x: [batch_size, in_ch, H, W]
        # temb: [batch_size, time_dim]

        h = self.resnet1(x, temb)
        # h: [batch_size, out_ch, H, W]

        h = self.attn(h)
        # h: [batch_size, out_ch, H, W]

        h = self.resnet2(h, temb)
        # h: [batch_size, out_ch, H, W]

        return h


class Up(nn.Module):
    def __init__(
        self, in_ch, out_ch, time_dim, heights_for_attn=(), add_upsampling=True
    ):
        super().__init__()
        # Apply attention only if input image's height (H) in listed in
        # `heights_for_attn`.
        self.heights_for_attn = heights_for_attn
        # in_ch is multiplied by 2 because input is concat of previous block's
        # output and matching `Down` block's output.
        self.resnets = nn.ModuleList(
            [
                ResNet(in_ch=2 * in_ch, out_ch=out_ch, time_dim=time_dim),
                ResNet(in_ch=out_ch, out_ch=out_ch, time_dim=time_dim),
            ]
        )
        self.attns = nn.ModuleList([Attn(in_ch=out_ch), Attn(in_ch=out_ch)])
        self.upsample = UpSample(out_ch) if add_upsampling else None

    def forward(self, x, res, temb):
        # x: [batch_size, in_ch, H, W]
        # temb: [batch_size, time_dim]

        # Concate across C (channel) dimension
        h = torch.cat((x, res), dim=1)
        # h: [batch_size, 2*in_ch, H, W]

        for resnet, attn in zip(self.resnets, self.attns):
            h = resnet(h, temb)
            # h: [batch_size, out_ch, H, W]

            if x.shape[2] in self.heights_for_attn:
                h = attn(h)
                # h: [batch_size, out_ch, H, W]

        if self.upsample:
            h = self.upsample(h)
            # h: [batch_size, out_ch, 2*H, 2*W]

        return h


class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        # x: [batch_size, in_ch, H, W]

        out = self.layers(x)
        # out : [batch_size, out_ch, H, W]

        return out


class UNet(nn.Module):
    TIME_EMBED_DIM = 128
    ORG_CHANNEL = 3
    # (in_ch, out_ch) pairs
    CHANNELS_DOWN_PATH = [(64, 64), (64, 128), (128, 256), (256, 512)]
    CHANNELS_UP_PATH = [(512, 256), (256, 128), (128, 64), (64, 64)]
    # Similar to the original paper, we add attn module to the (Down/Up) blocks
    # whose input tensors has height of 16.
    HEIGHTS_FOR_ATTN_RESOLUTION = (16,)

    def __init__(self):
        super().__init__()
        # number of block in Down/Up path.
        num_resolutions = len(UNet.CHANNELS_DOWN_PATH)
        self.tembeding = TimeEmbedding(UNet.TIME_EMBED_DIM)
        self.pre = Pre(in_ch=UNet.ORG_CHANNEL, out_ch=UNet.CHANNELS_DOWN_PATH[0][0])
        self.downs = nn.ModuleList([])
        for idx, (in_ch, out_ch) in enumerate(UNet.CHANNELS_DOWN_PATH):
            self.downs.append(
                Down(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    time_dim=UNet.TIME_EMBED_DIM,
                    heights_for_attn=UNet.HEIGHTS_FOR_ATTN_RESOLUTION,
                    add_downsampling=idx != num_resolutions - 1,
                )
            )
        self.middle = Middle(
            in_ch=UNet.CHANNELS_DOWN_PATH[-1][1],
            out_ch=UNet.CHANNELS_DOWN_PATH[-1][1],
            time_dim=UNet.TIME_EMBED_DIM,
        )
        self.ups = nn.ModuleList([])
        for idx, (in_ch, out_ch) in enumerate(UNet.CHANNELS_UP_PATH):
            self.ups.append(
                Up(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    time_dim=UNet.TIME_EMBED_DIM,
                    heights_for_attn=UNet.HEIGHTS_FOR_ATTN_RESOLUTION,
                    add_upsampling=idx != 0,
                )
            )
        self.out = Out(in_ch=UNet.CHANNELS_UP_PATH[-1][1], out_ch=UNet.ORG_CHANNEL)

    def forward(self, x, timesteps):
        # x: [batch_size, C, H, W]
        # timesteps: [batch_size, 1]

        # Ensure x and timesteps are on same device.
        timesteps.to(x.device)

        temb = self.tembeding(timesteps)
        # temb: [batch_size, TIME_EMBED_DIM]

        h = self.pre(x)
        # h: [batch_size, CHANNELS_DOWN_PATH[0][0], H, W]

        res = [h]
        for down in self.downs:
            res.append(down(res[-1], temb))

        h = res[-1]
        # let n = len(CHANNELS_DOWN_PATH) - 1 # -1 b/c last block doesnot downsample.
        # h: [batch_size, CHANNELS_DOWN_PATH[-1][1], H/ n, W/n]

        h = self.middle(res[-1], temb)
        # h: [batch_size, CHANNELS_DOWN_PATH[-1][1], H/ n, W/n]

        for up in self.ups:
            h = up(h, res.pop(), temb)
        # h: [batch_size, CHANNELS_UP_PATH[-1][1], H, W]

        out = self.out(h)
        # out: [batch_size, C, H, W]

        return out
