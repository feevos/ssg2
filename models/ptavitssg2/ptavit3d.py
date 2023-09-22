# Modified with my attention from https://github.com/lucidrains/vit-pytorch

from functools import partial
import numpy as np

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class LayerNormChannelsLast3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        x = rearrange(input,'b c s h w-> b s h w c')
        x = self.norm(x)
        x = rearrange(x,'b s h w c -> b c s h w')
        return x

class PreNormResidual3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PreNormResidualAtt3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return (self.fn(self.norm(x)) + 1)*x

class FeedForward3D(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = rearrange(x,'b c s h w -> b s h w c')
        x = self.net(x)
        return rearrange(x,'b s h w c -> b c s h w')

# MBConv
class SqueezeExcitation3D(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c s h w -> b s c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b s c -> b c s 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual3D(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)
    




def MBConv3D(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv3d(dim_in, hidden_dim, 1),
        #CustomBatchNorm3d(hidden_dim),
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        nn.Conv3d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        #CustomBatchNorm3d(hidden_dim),
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation3D(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv3d(hidden_dim, dim_out, 1),
        #CustomBatchNorm3d(dim_out)
        nn.BatchNorm3d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual3D(net, dropout = dropout)

    return net

# attention related classes
from ssg2.nn.layers.ptattention3d import * 
class PTAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        nheads = 32,
        dropout = 0.,
        scales = (4,4),
        verbose=False):
        super().__init__()

        if verbose:
            print("nfilters::{}, scales::{}, nheads::{}".format(dim, scales,nheads))


        self.att      =  RelPatchAttentionTHW(
                                in_channels  = dim,
                                out_channels = dim,
                                nheads       = nheads,
                                scales       = scales,
                                norm         = 'GroupNorm',
                                norm_groups  = dim//4)

    def forward(self,input):       
        return   self.att(input,input).permute([0,2,1,3,4]) # [-0.1,0.1] with d2sigmoid


class PTAViTStage3D_no_down(nn.Module):
    def __init__(
        self,
        layer_dim_in,
        layer_dim,
        layer_depth,
        nheads,
        scales,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        norm_type='BatchNorm',
        norm_groups=None
    ):
        super().__init__()


        stage = []
        for stage_ind in range(layer_depth):
            # This gives nans, don't know why at the moment, getting there ...               
            block = nn.Sequential(
                MBConv3D(
                    layer_dim_in, 
                    layer_dim_in,
                    downsample = False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                PreNormResidualAtt3D(
                    layer_dim_in, 
                    PTAttention3D(
                        dim = layer_dim_in, 
                        nheads = nheads, 
                        dropout = dropout,
                        scales=scales)),
                PreNormResidual3D(
                    layer_dim_in, 
                    FeedForward3D(
                        dim = layer_dim_in, 
                        dropout = dropout))
            )


            stage.append(block)
        self.stage = torch.nn.Sequential(*stage)
    def forward(self,input):
        return self.stage(input)


