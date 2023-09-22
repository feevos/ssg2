import torch
from ssg2.models.unet.ptavit_dn_features   import * 
from ssg2.models.unet.head_cmtsk           import * 




class ptavit_dn_cmtsk(torch.nn.Module):
    def __init__(self, in_channels, NClasses,  nfilters_init=96, nheads_start=96//4, depths=[2,2,5,2], spatial_size_init=256, verbose=True, norm_type='GroupNorm', norm_groups=4, correlation_method='linear',segm_act='sigmoid'):
        super().__init__()
                   
        self.features = ptavit_dn_features(in_channels = in_channels,  spatial_size_init=spatial_size_init, nfilters_init=nfilters_init, nheads_start = nheads_start, depths = depths, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups, correlation_method=correlation_method)

        # Final Segmentation head units 
        self.head = head_cmtsk(nfilters=nfilters_init,NClasses=NClasses, norm_type=norm_type,norm_groups=norm_groups)



    def forward(self,input):
        features = self.features(input)
        outs = self.head(features)
        return outs 







