import torch
import numpy as np

from ssg2.nn.layers.conv2Dnormed import *
from ssg2.nn.layers.patchfusion import  *
from ssg2.nn.layers.ptattention import  *  
from ssg2.nn.layers.scale import *
from ssg2.nn.layers.combine import *
from ssg2.nn.units.ptavit import *


class FusionCAT(torch.nn.Module):
    def __init__(self,nfilters_in,nfilters_out, nheads, kernel_size=3, padding=1, norm = 'BatchNorm', norm_groups=None):
        super().__init__()
        
        self.fuse = Conv2DNormed(in_channels=nfilters_in*2, out_channels=nfilters_out,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)

    def forward(self, out12, out21):

        fuse = self.fuse(torch.cat([out12, out21],dim=1))
        fuse = torch.relu(fuse)

        return fuse



class FuseHiLo(torch.nn.Module):
    # BC: Balanced (features) Crisp (boundaries) 
    def __init__(self, nfilters, spatial_size=256,scales=(4,8),   norm_type = 'BatchNorm', norm_groups=None):
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
        #print(convl.shape)
        conv = self.conv2d(convl)
        conv = torch.relu(conv)

        # Apply attention
        conv = conv * (1.+self.att(conv))

        return conv




class mantis_ca_dn_features(torch.nn.Module):
    # This is a modification of the mantis architecture, developed in Diakogiannis et al 2021 https://www.mdpi.com/2072-4292/13/18/3707
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4, correlation_method='linear'):
        super().__init__()

        def closest_power_of_2(num_array):
            log2_array = np.log2(num_array)
            rounded_log2_array = np.round(log2_array)
            closest_power_of_2_array = np.power(2, rounded_log2_array)
            return np.maximum(closest_power_of_2_array, 1).astype(int)


        def resize_scales(channel_size, spatial_size, scales_all):
            temp = np.array(scales_all)*np.array([channel_size/96,spatial_size/256])
            return closest_power_of_2(temp).tolist()

        scales_all = [[16,16],[32,8],[64,4],[128,2],[128,2],[128,1],[256,1],[256,1]] # DEFAULT, nice results 

        self.depth = depth = len(depths)
        num_stages = len(depths)
        dims = tuple(map(lambda i: (2 ** i) * nfilters_init, range(num_stages)))
        dims = (nfilters_init, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.conv1     = Conv2DNormed(in_channels=in_channels, out_channels = nfilters_init, kernel_size=1,padding=0,strides=1, norm_type=norm_type, num_groups=norm_groups)
        self.fuse_first = FusionV2(nfilters_in=nfilters_init, nfilters_out= nfilters_init, spatial_size=spatial_size_init, scales = scales_all[0], 
                correlation_method=correlation_method,
                norm=norm_type, norm_groups=norm_groups)
        
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
        self.fuse     = [] 
        self.atts_fuse     = [] 

        if verbose:
            print (" @@@@@@@@@@@@@ Going DOWN @@@@@@@@@@@@@@@@@@@ ")
        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
        #for idx,  layer_depth in enumerate(depths):
            nheads = nheads_start * 2**idx #
            scales = scales_all[idx]
            spatial_size = spatial_size_init // 2**idx

            #print ("drop_path istart::{}, iend::{}".format(dp_istart,dp_iend))
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

            self.fuse.append( FusionCAT( 
                nfilters_in         =   layer_dim, 
                nfilters_out        =   layer_dim,  
                nheads              =   nheads,
                norm                =   norm_type, 
                norm_groups         =   norm_groups)  
                )

            self.atts_fuse.append( RelPatchAttention2D(
                in_channels         =   layer_dim, 
                out_channels        =   layer_dim, 
                kernel_size         =   3, 
                padding             =   1, 
                nheads              =   nheads, 
                norm                =   norm_type, 
                norm_groups         =   norm_groups, 
                spatial_size        =   spatial_size, 
                scales              =   scales, 
                correlation_method  =   correlation_method)
                )


        self.stages_dn = torch.nn.ModuleList(self.stages_dn)
        self.fuse     = torch.nn.ModuleList(self.fuse)
        self.atts_fuse     = torch.nn.ModuleList(self.atts_fuse)


        self.stages_up = [] 
        self.UpCombs  = [] 

        # Reverse order, ditch first 
        dim_pairs = dim_pairs[::-1]
        depths    = depths[::-1]
        
        dim_pairs = dim_pairs[:-1]
        #depths    = depths[:-1]# works 
        depths    = depths[1:]

        

        if verbose:
            print (" XXXXXXXXXXXXXXXXXXXXX Coming up XXXXXXXXXXXXXXXXXXXXXXXXX " )

        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            idx = len(depths)-1 - idx
            nheads = int(nheads_start * 2**(idx)) 
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

    def forward(self, input_t1, input_t2):

        
        conv1_t1 = self.conv1(input_t1)
        conv1_t2 = self.conv1(input_t2)
            
        fuse1_hi_res = self.fuse_first(conv1_t1,conv1_t2)
        
	# Reduce spatial size 
        conv1 = self.conv_stem(conv1_t1)
        conv2 = self.conv_stem(conv1_t2)
        
        # ******** Going down ***************
        #print ("Going DOWN")
        fusions   = []
        for idx in range(self.depth):
            #print ("Down index::{}".format(idx))
            #print ("Shapes BEFORE stages::{}, {}".format(conv1.shape, conv2.shape))
            conv1 = self.stages_dn[idx](conv1)
            conv2 = self.stages_dn[idx](conv2)

            conv1 = conv1 + self.atts_fuse[idx](conv2,conv1)
            conv2 = conv2 + self.atts_fuse[idx](conv1,conv2)


            # Evaluate fusions 
            fusions = fusions + [self.fuse[idx](conv1,conv2)]

        # ******* Coming up ****************
        convs_up = fusions[-1]
        convs_up = torch.relu(convs_up) 
        for idx in range(self.depth-1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            convs_up = self.stages_up[idx](convs_up)
         


        final = self.fuse_hi_lo(convs_up, fuse1_hi_res)
        return final 
    

