import torch

from ssg2.models.ptavitssg2.mantis_ca_dn_features import * 

class ptavitssg2_dn_features(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4,correlation_method='linear'):
        super().__init__()
 


        self.features = mantis_ca_dn_features(
                in_channels=in_channels, 
                spatial_size_init=spatial_size_init, 
                nfilters_init=nfilters_init, 
                nheads_start=nheads_start, 
                depths=depths, 
                verbose=verbose, 
                norm_type=norm_type, 
                norm_groups=norm_groups,
                correlation_method=correlation_method)
       

    def forward(self, input_t1, input_t2):
        # input_t1.shape --> b x c x h x w 
        # input_t2.shape --> b x c x SEQUENCE x h x w
        
        # Now get into the sequence of events 
        features = [] # A list of all outs 
        for seq_idx in range(input_t2.shape[2]):
            tinput_t2 =  input_t2[:,:,seq_idx]
            #print(input_t1.shape, tinput_t2.shape)
            tout = self.features(input_t1,tinput_t2).unsqueeze(2)
            features.append(tout)

        return torch.cat(features,dim=2)



