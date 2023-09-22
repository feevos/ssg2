import torch
#from trchprosthesis.experimental.models.mantis_v3.head_2Dnext           import * 
from trchprosthesis.experimental.models.mantis_triad_ca_fzcorr.head_cmtsk_3df  import * 
from trchprosthesis.experimental.models.mantis_triad_ca_fzcorr.mantis_triad_ca_dn_features import *  # same as v3, but no swap 


# Mantis conditioned multitasking. 
class ssg2_dn_cmtsk(torch.nn.Module):
    def __init__(self, in_channels, NClasses,  nfilters_init=96, nfilters_embed=96, nheads_start=96//4, depths=[2,2,5,2], spatial_size_init=256, verbose=True, norm_type='GroupNorm', norm_groups=4, metric_learning=False, correlation_method='linear',nassociations=None,representation='1Hot',segm_act='sigmoid', nresblocks=2,model3dtype ='PatchMaxVit3D' ):
        super().__init__()
        
        self.features = mantis_triad_ca_dn_features(in_channels = in_channels,  spatial_size_init=spatial_size_init, nfilters_init=nfilters_init, nfilters_embed =  nfilters_embed, nheads_start = nheads_start, depths = depths, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups, metric_learning=metric_learning, correlation_method=correlation_method,nassociations=nassociations)
        # Segmentation head 
        self.head = head_cmtsk_3D(nfilters=nfilters_embed,spatial_size = spatial_size_init, NClasses=NClasses, norm_type=norm_type,norm_groups=norm_groups,representation=representation,segm_act=segm_act, nresblocks = nresblocks, model3dtype = model3dtype)

    def forward_inference(self,input_t1, input_t2):
        lst_out = self.features(input_t1,input_t2)
        #print (lst_out.shape)
        return self.head.forward_inference(lst_out)

    def forward(self,input_t1, input_t2):
        lst_out = self.features(input_t1,input_t2)
        #print (lst_out.shape)
        return self.head(lst_out)





