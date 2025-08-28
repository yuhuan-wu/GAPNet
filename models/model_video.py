import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import time
from models.utils import ConvBNReLU, ReceptiveVit
from models.vgg import vgg16
from models.resnet import resnet50, resnet101, resnet152, Bottleneck
from models.MobileNetV2 import mobilenetv2
# try:
from models.vit_fusion import Block
from models.vit_fusion import BlockEA as FusionEA


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class FuseNet(nn.Module):
    def __init__(self, inchannels=320):
        super(FuseNet, self).__init__()
        self.d_conv1 = InvertedResidual(inchannels, inchannels, residual=True)
        self.d_linear = nn.Sequential(
            nn.Linear(inchannels, inchannels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(inchannels, inchannels, bias=True),
        )
        self.d_conv2 = InvertedResidual(inchannels, inchannels, residual=True)

    def forward(self, x, x_d):
        x_f = self.d_conv1(x * x_d)
        x_d1 = self.d_linear(x.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3)
        x_f1 = self.d_conv2(torch.sigmoid(x_d1) * x_f * x_d)
        return x_f1

class FuseGPC(nn.Module):
    def __init__(self, inchannels=32, outchannels=32):
        super(FuseGPC, self).__init__()
        self.fuse1 = ReceptiveVit(inchannels*2, outchannels)
        self.fuse2 = ReceptiveVit(outchannels, outchannels)
    
    def forward(self, x1, x2):
        x_att = x1 * x2 + x1 + x2
        x = self.fuse1(torch.cat((x1, x_att), dim=1))
        return self.fuse2(x)

class GAPNet(nn.Module):
    def __init__(self, arch='mobilenetv2', pretrained=True,
                 enc_channels=[64, 128, 256, 512, 512, 256, 256],
                 dec_channels=[32, 64, 128, 128, 256, 256, 256], freeze_s1=False,
                 last_channel=80, global_guidance=True, diverse_supervision=True):
        super(GAPNet, self).__init__()
        
        self.arch = arch
        self.backbone = eval(arch)(pretrained)
        self.backbone_1 = eval(arch)(pretrained)
        self.global_guidance = global_guidance
        self.diverse_supervision = diverse_supervision

        # if 'mobilenetv2' in arch:
        enc_channels = [16, 24, 32, 96, 160, last_channel]
        dec_channels = [16, 40, 40, 40, 40, 40]

        use_dwconv = 'mobilenet' in arch

        self.multi_fusions = nn.ModuleList(
            [
             FuseGPC(enc_channels[2], enc_channels[2]),
             FusionEA(enc_channels[3], dwconv=False),
             FusionEA(enc_channels[4], dwconv=False),
            ]
        )

        
        self.vit_global = nn.ModuleList([Block(dim=enc_channels[i+4], out_features=enc_channels[-1]) for i in range(2)])

        self.fpn = TransformerDecoder(enc_channels, dec_channels, rec_dwconv=use_dwconv)

        self.cls1 = nn.Conv2d(1*dec_channels[-1], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[-1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(enc_channels[-1], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[-1], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(1*dec_channels[-1], 1, 1, stride=1, padding=0)
        self.cls6 = nn.Conv2d(dec_channels[-1], 1, 1, stride=1, padding=0)
        

        self._freeze_backbone(freeze_s1=freeze_s1)

    def _freeze_backbone(self, freeze_s1):
        if not freeze_s1:
            return
        assert ('resnet' in self.arch and '3x3' not in self.arch)
        m = [self.backbone.conv1, self.backbone.bn1, self.backbone.relu]
        print("freeze stage 0 of resnet")
        for p in m:
            for pp in p.parameters():
                p.requires_grad = False

    def forward(self, input, input_1=None):
        start_time = time.time()
        
        # Backbone
        backbone_features = self.backbone(input)

        # Multi-Modal or Video
        if input_1 is not None:
            supp_features = self.backbone_1(input_1)
            #backbone_features[1] = self.multi_fusions[0](backbone_features[1], supp_features[1])
            backbone_features[2] = self.multi_fusions[0](backbone_features[2], supp_features[2])
            backbone_features[3] = self.multi_fusions[1]([backbone_features[3], supp_features[3]])
            backbone_features[4] = self.multi_fusions[2]([backbone_features[4], supp_features[4]])

        
        # Global guidance
        global_features = backbone_features[-1]
        for idx, blk in enumerate(self.vit_global):
            B, C, H, W = global_features.size()
            global_features = global_features.reshape(B, C, H*W).permute(0, 2, 1)
            global_features = blk(global_features, H, W)
            global_features = global_features.permute(0, 2, 1).reshape(B, -1, H, W)

        
        # FPN
        features = self.fpn(backbone_features + [global_features])
        
        # Saliency maps
        saliency_maps = []
        for idx, feature in enumerate(features):
            saliency_maps.append(F.interpolate(
                getattr(self, 'cls' + str(idx + 1))(feature),
                input.shape[2:],
                mode='bilinear',
                align_corners=False)
            )
        
        # Total time
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        
        # Print timing statistics every 10 iterations (using a static counter)
        if not hasattr(GAPNet.forward, 'counter'):
            GAPNet.forward.counter = 0
        GAPNet.forward.counter += 1
        
        return torch.sigmoid(torch.cat(saliency_maps, dim=1))


class TransformerDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, rec_dwconv=True, vit_dwconv=True, diverse_supervision=True,
                 rec_ratio=[1/8, 1/8, 2/8]):
        super(TransformerDecoder, self).__init__()
        
        # self.low_scale = low_scale
        self.inners_a = nn.ModuleList()
        self.diverse_supervision = diverse_supervision
        
        for i in range(3):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i]//2, ksize=1, pad=0))
        for i in range(3, len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i], ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        
        self.fuse = nn.ModuleList()
        # low-mid, mid-high feature fusion
        dilation = [[1, 2, 4, 6]] * (len(in_channels) - 4) + [[1, 2, 3, 4]] * 2 + [[1, 1, 1, 1]] * 2
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5

        i = -5
        self.fuse.append(nn.Sequential(
            ReceptiveVit(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
                         dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio),
            ReceptiveVit(out_channels[i], out_channels[-1], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio)))
        
        self.fuse.append(nn.Sequential(
                ReceptiveVit(2*out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio),
                ReceptiveVit(out_channels[i], out_channels[-1], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio)))


        i = -3
        
        self.fuse.append(FusionEA(dim=out_channels[-1], dwconv=vit_dwconv))
        self.fuse.append(FusionEA(dim=out_channels[-1], dwconv=vit_dwconv))

        i = -1
        self.fuse.insert(0, nn.Identity())

        self.fuse.append(nn.Sequential(
            ReceptiveVit(2*out_channels[-1], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio),
            ReceptiveVit(out_channels[i], out_channels[-1], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=rec_dwconv, rec_ratio=rec_ratio)))

    def forward(self, features):
        results = []

        # Obtain middle/high features
        idxs = [-3, -2]
        inner_high_low = self.inners_a[idxs[0]](features[idxs[0]])
        inner_high_high = self.inners_a[idxs[1]](features[idxs[1]])
        mid_high_result = self.fuse[3]([inner_high_low, inner_high_high])
        
        # Obtain low/middle features
        idxs = [-5, -4]
        inner_low_low = self.inners_a[idxs[0]](features[idxs[0]])
        inner_low_high = self.inners_a[idxs[1]](features[idxs[1]])
        low_size = inner_low_low.shape[2:]
        inner_low_high = F.interpolate(inner_low_high,
                                       size=low_size,
                                       mode='bilinear',
                                       align_corners=False)
        low_mid_edge = self.fuse[1](torch.cat((inner_low_low, inner_low_high), dim=1))

        results.append(low_mid_edge)

        global_center = features[-1]
        results.append(global_center)

        # Mid-high global fusion
        mid_high_global = self.fuse[4]([mid_high_result, self.inners_a[-1](global_center)])
        
        results.append(mid_high_global)

        # Low-mid global fusion (EA only)
        global_upsampled = F.interpolate(self.inners_a[-1](global_center),
                                             size=low_size,
                                             mode='bilinear',
                                             align_corners=False)
        low_mid_global = self.fuse[2](torch.cat((low_mid_edge, global_upsampled), dim=1))

        results.append(low_mid_global)

        # Full result
        mid_high_global = F.interpolate(mid_high_global,
                                        size=inner_low_low.shape[2:],
                                        mode='bilinear',
                                        align_corners=False)
        full_result = self.fuse[5](torch.cat((low_mid_global, mid_high_global), dim=1))

        results.insert(0, full_result)
        results.append(mid_high_result)
        
        if self.diverse_supervision:
            return results
        else:
            return results[:1]

