import torch

from ssg2.models.heads.head_cmtsk_3df  import * 
from ssg2.models.ptavitssg2.ptavitssg2_dn_features import *  


# Conditioned multitasking. 
class ptavitssg2_dn_cmtsk(torch.nn.Module):
    def __init__(self, in_channels, NClasses,  nfilters_init=96, nheads_start=96//4, depths=[2,2,5,2], spatial_size_init=256, verbose=True, norm_type='GroupNorm', norm_groups=4, correlation_method='linear',segm_act='sigmoid', nblocks3D=2):
        super().__init__()
        
        self.features = ptavitssg2_dn_features(in_channels = in_channels,  spatial_size_init=spatial_size_init, nfilters_init=nfilters_init, nheads_start = nheads_start, depths = depths, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups, correlation_method=correlation_method)

        # Segmentation head 
        self.head = head_cmtsk_3D(nfilters=nfilters_init,spatial_size = spatial_size_init, NClasses=NClasses, norm_type=norm_type,norm_groups=norm_groups,segm_act=segm_act, nblocks3D = nblocks3D)

    def forward_inference(self,input_t1, input_t2):
        # Computationally less expensive option 
        # Outputs only Target predictions 
        lst_out = self.features(input_t1,input_t2)
        
        return self.head.forward_inference(lst_out)

    def forward(self,input_t1, input_t2):
        lst_out = self.features(input_t1,input_t2)
        
        return self.head(lst_out)





