import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from carafe import carafe_naive


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, use_bn=True, frozen=False, residual=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                                        dilation=1, groups=groups, bias=bias,
                                        use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, self.num_heads, 3, C // self.num_heads).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, NUM_HEADS, SEQ_LEN, CHANNELS_PER_HEAD

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.mlp(x)
        return x


class LRSA(nn.Module):
    def __init__(self, dim, m, norm_layer=nn.LayerNorm):
        super(LRSA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((m, m))
        self.sa = SelfAttention(dim=dim)
        self.norm = norm_layer(dim)

    def forward(self, x):
        img_size = x.size()[-2:]
        x = self.pool(x)
        B, C, H, W = x.size()
        # print(x.size())
        x = x.reshape(B, C, H*W).transpose(1, 2)
        x = self.sa(self.norm(x))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = F.interpolate(x, img_size, mode='bilinear', align_corners=False)
        return x


class ReceptiveVit(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None, use_dwconv=False, rec_ratio=[1/8, 1/8, 2/8], m=7):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveVit, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        # self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale
        # print(rec_ratio)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        scpc_width = [int(math.floor(a*self.width*scale)) for a in rec_ratio]
        scpc_width.append(self.width*scale - sum(scpc_width))

        for i in range(self.nums):
            if use_dwconv:
                self.convs.append(nn.Conv2d(scpc_width[i], scpc_width[i], kernel_size=3,
                                            padding=dilation[i], dilation=dilation[i], groups=scpc_width[i], bias=False))
            else:
                self.convs.append(nn.Conv2d(scpc_width[i], scpc_width[i], kernel_size=3,
                                            padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(scpc_width[i]))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        # self.aggregation = (len(set(rec_ratio)) == 1)
        self.aggregation = False
        self.scpc = scpc_width
        self.m = m
        if m > 0:
            self.lrsa = LRSA(dim=inplanes, m=m) if inplanes == planes else nn.Identity()
        # print(scpc_width)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = []
        for i in range(self.nums):
            index = range(self.scpc[i]) if i == 0 else range(sum(self.scpc[:i]), sum(self.scpc[:i+1]))
            # print(index)
            # print(x.size())
            spx.append(out[:, index, :, :])

        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if out.size() == x.size():
            out = out + x

        if self.m > 0:
            attention = self.lrsa(x)
            if out.size() == attention.size():
                out = out + attention

        # else:
        #     print(out.size(), x.size(), attention.size())
        out = self.relu(out)

        return out
