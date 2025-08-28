from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import numpy as np


class FFN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, act_layer=nn.Hardswish, drop=0., dwconv=False):
        super().__init__()
        self.out_features = out_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = dwconv
        if dwconv:
            self.dwconv = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features, bias=False, dilation=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print()
        x = x.reshape(B, H, W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        if self.dwconv:
            x = x + self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, self.out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0,2,1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.reshape(B, self.out_features, -1).permute(0,2,1)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, proj_drop=0.):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, self.num_heads, 3, C // self.num_heads).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B, NUM_HEADS, SEQ_LEN, CHANNELS_PER_HEAD
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, out_features, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
    
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, proj_drop=drop)

        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim,out_features=out_features, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop, ksize=3)
    
    def forward(self, x, H, W, d_convs=None):
        # print(x.size())
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        if self.dim == self.out_features:
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        else:
            x = self.mlp(self.norm2(x), H, W)

        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # self.attn_drop = nn.Dropout(attn_drop)

        # self.pooling = nn.AdaptiveAvgPool1d(N2)
        self.mlp = nn.Linear(dim, dim)
        # self.proj = nn.Linear(N1, N2)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, q_conc=True):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape

        q = self.proj_q(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = self.proj_kv(x2).reshape(B2, N2, self.num_heads, 2, C2 // self.num_heads).permute(3, 0, 2, 1, 4)
        k, v = kv[0], kv[1]

        if q_conc:
            N1 -= N2

        q = q[:, :, :N1, :]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B1, N1, C1)
        # x = self.pooling(x.transpose(1,2))
        x = self.mlp(x)

        return x

# efficient transformer


class BlockEA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 drop_path=0., norm_layer=nn.LayerNorm, dwconv=True, expand=1):
        super().__init__()
        self.dim = dim
        # self.out_features = out_features
        self.norm1 = norm_layer(dim)

        self.attn = EfficientAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ffn = FFN(in_features=dim, out_features=dim * expand, dwconv=dwconv)

    def forward(self, xs):
        x1, x2 = xs
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        x1 = x1.reshape(B1, C1, H1*W1).transpose(1, 2)
        x2 = x2.reshape(B2, C2, H2*W2).transpose(1, 2)
        
        x1 = torch.cat((x1, x2), dim=1)

        x = self.attn(self.norm1(x1), self.norm1(x2))
        x = x + self.ffn(self.norm2(x), H1, W1)

        return x.transpose(1, 2).reshape(B1, C1, H1, W1)
