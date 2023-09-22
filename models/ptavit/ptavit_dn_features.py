import torch
import numpy as np

from ssg2.nn.layers.conv2Dnormed import *
from ssg2.nn.layers.scale import *
from ssg2.nn.layers.combine import *
from ssg2.nn.layers.ptattention import *
from ssg2.nn.units.ptavit import *


class FuseHiLo(torch.nn.Module):
    def __init__(self, nfilters,  spatial_size=256,scales=(4,8),   norm_type = 'BatchNorm', norm_groups=None):
        super().__init__()
        
        self.embedding1 = Conv2DNormed(in_channels = nfilters, out_channels = nfilters, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)
        self.embedding2 = Conv2DNormed(in_channels = nfilters, out_channels = nfilters, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)


        self.upscale = UpSample2D(in_channels=nfilters,out_channels=nfilters,scale_factor=4,norm_type=norm_type,norm_groups=norm_groups)


        self.conv2d = Conv2DNormed(in_channels=nfilters*2, out_channels = nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.att = PatchAttention2D( in_channels=nfilters, out_channels = nfilters,nheads=nfilters//4,norm=norm_type,norm_groups=norm_groups,
                spatial_size=spatial_size,
                scales=scales,
                correlation_method='linear')


    def forward(self, UpConv4, conv1):
        # conv1: full resolution
        # UpConv4: 1/4 of original resolution 

        UpConv4 = self.embedding1(UpConv4)
        UpConv4 = self.upscale(UpConv4)
        conv1   = self.embedding2(conv1)
        
        # second last layer 
        convl = torch.cat([conv1,UpConv4],dim=1)
        conv = self.conv2d(convl)
        conv = torch.relu(conv)

        # Apply attention
        conv = conv * (1.+self.att(conv))

        return conv




class ptavit_dn_features(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4,correlation_method='linear'):
        super().__init__()
 

        def closest_power_of_2(num_array):
            log2_array = np.log2(num_array)
            rounded_log2_array = np.round(log2_array)
            closest_power_of_2_array = np.power(2, rounded_log2_array)
            return np.maximum(closest_power_of_2_array, 1).astype(int)


        def resize_scales(channel_size, spatial_size, scales_all):
            temp = np.array(scales_all)*np.array([channel_size/96,spatial_size/256])
            return closest_power_of_2(temp).tolist()


        scales_all = [[16,16],[32,8],[64,4],[128,2],[128,2],[128,1],[256,1],[256,1]] # DEFAULT, nice results for 256 image patches 
            


        self.depth = depth = len(depths)
        num_stages = len(depths)
        dims = tuple(map(lambda i: (2 ** i) * nfilters_init, range(num_stages)))
        dims = (nfilters_init, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.conv1     = Conv2DNormed(in_channels=in_channels, out_channels = nfilters_init, kernel_size=1,padding=0,strides=1, norm_type=norm_type, num_groups=norm_groups)
        
        # Scale 1/2 
        self.conv_stem = nn.Sequential(
            nn.Conv2d( nfilters_init, nfilters_init, 3, stride = 2, padding = 1),
            nn.Conv2d(nfilters_init, nfilters_init, 3, padding = 1)
            )

        # The stem scales 1/2, the MaxViTFD layers scale 1/2 in first MBConv
        spatial_size_init = spatial_size_init//4
        scales_all = resize_scales(nfilters_init, spatial_size_init,scales_all)
        self.scales_all = scales_all 

        # List of convolutions and pooling operators 
        self.stages_dn = [] 

        if verbose:
            print (" @@@@@@@@@@@@@ Going DOWN @@@@@@@@@@@@@@@@@@@ ")
        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            nheads = nheads_start * 2**idx #
            
            scales = scales_all[idx]
            spatial_size = spatial_size_init // 2**idx

            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(idx,layer_dim_in,layer_dim,layer_depth,spatial_size, scales)) 

            self.stages_dn.append(PTAViTStage(
                layer_dim_in=layer_dim_in,
                layer_dim=layer_dim,
                layer_depth=layer_depth,
                nheads=nheads,
                scales=scales,
                spatial_size=spatial_size,
                downsample=True,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,
                correlation_method=correlation_method
                ))



        self.stages_dn = torch.nn.ModuleList(self.stages_dn)


        self.stages_up = [] 
        self.UpCombs  = [] 

        # Reverse order, ditch first 
        dim_pairs = dim_pairs[::-1]
        depths    = depths[::-1]
        
        dim_pairs = dim_pairs[:-1]
        depths    = depths[1:]

        

        if verbose:
            print (" XXXXXXXXXXXXXXXXXXXXX Coming up XXXXXXXXXXXXXXXXXXXXXXXXX " )

        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            idx = len(depths)-1 - idx
            nheads = int(nheads_start * 2**(idx)) #
            spatial_size = spatial_size_init // 2**(idx)
            scales = scales_all[idx]

            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(2*depth-idx-2, 
                    layer_dim_in, layer_dim_in, layer_depth,spatial_size, scales))



            self.stages_up.append(PTAViTStage(
                layer_dim_in=layer_dim_in,
                layer_dim=layer_dim_in,
                layer_depth=layer_depth,
                nheads=nheads,
                scales=scales,
                spatial_size=spatial_size,
                downsample=False,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,
                correlation_method=correlation_method
                ))


            self.UpCombs.append(combine_layers(
                layer_dim_in, 
                norm_type=norm_type,
                norm_groups=norm_groups))

        
        self.stages_up   = torch.nn.ModuleList(self.stages_up)
        self.UpCombs    = torch.nn.ModuleList(self.UpCombs)


        self.fuse_hi_lo = FuseHiLo( nfilters=layer_dim_in, spatial_size=spatial_size,scales=(4,8),   norm_type = norm_type, norm_groups=norm_groups)



    def forward(self, input):
        # input.shape --> B x C x H x W 
        
        conv1_hi_res = self.conv1(input)
            
        
	# Reduce spatial size 
        conv1 = self.conv_stem(conv1_hi_res)
        
        # ******** Going down ***************
        fusions   = []
        for idx in range(self.depth):
            conv1 = self.stages_dn[idx](conv1)
            fusions = fusions + [conv1]

        # ******* Coming up ****************
        convs_up = fusions[-1]
        convs_up = torch.relu(convs_up) # middle of network activation
        for idx in range(self.depth-1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            convs_up = self.stages_up[idx](convs_up)

        # Fuse original spatial size with 1/4 of it, to regain detail in layers
        final = self.fuse_hi_lo(convs_up, conv1_hi_res)
        return final 
    

