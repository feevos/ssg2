import torch

from ssg2.models.ssg2.mantis_ca_dn_features import * 

"""
This model uses identical features as mantis_v2 but it consumes a list of input images for target 2 
"""
class mantis_triad_ca_dn_features(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nfilters_embed=96, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4,metric_learning=False,correlation_method='linear',nassociations=None):
        super().__init__()
 


        self.features = mantis_ca_dn_features(
                in_channels=in_channels, 
                spatial_size_init=spatial_size_init, 
                nfilters_init=nfilters_init, 
                nfilters_embed=nfilters_embed, 
                nheads_start=nheads_start, 
                depths=depths, 
                verbose=verbose, 
                norm_type=norm_type, 
                norm_groups=norm_groups,
                metric_learning=metric_learning,
                correlation_method=correlation_method,
                nassociations=nassociations)
       

    def forward(self, input_t1, input_t2):
        # input_t1.shape --> b x c x h x w 
        # input_t2.shape --> b x SEQUENCE x c x h x w
        
        # Now get into the sequence of events 
        features = [] # A list of all outs 
        for seq_idx in range(input_t2.shape[1]):
            tinput_t2 =  input_t2[:,seq_idx]
            #print(input_t1.shape, tinput_t2.shape)
            tout = self.features(input_t1,tinput_t2).unsqueeze(1)
            features.append(tout)

        return torch.cat(features,dim=1)



import threading
class mantis_triad_ca_dn_features_thread(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nfilters_embed=96, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4,metric_learning=False,correlation_method='linear',nassociations=None):
        super().__init__()
 


        self.features = mantis_ca_dn_features(
                in_channels=in_channels, 
                spatial_size_init=spatial_size_init, 
                nfilters_init=nfilters_init, 
                nfilters_embed=nfilters_embed, 
                nheads_start=nheads_start, 
                depths=depths, 
                verbose=verbose, 
                norm_type=norm_type, 
                norm_groups=norm_groups,
                metric_learning=metric_learning,
                correlation_method=correlation_method,
                nassociations=nassociations)

    def _async_features(self, input_t1, input_t2, features, idx):
        tout = self.features(input_t1, input_t2).unsqueeze(2)
        features[idx] = tout

    def forward(self, input_t1, input_t2):
        # input_t1.shape --> b x c x h x w 
        # input_t2.shape --> b x SEQUENCE x c x h x w

        # Now get into the sequence of events 
        features = [None] * input_t2.shape[1] # A list of all outs
        threads = []

        for seq_idx in range(input_t2.shape[1]):
            tinput_t2 = input_t2[:, seq_idx]
            print(input_t1.shape, tinput_t2.shape)
            thread = threading.Thread(target=self._async_features, args=(input_t1, tinput_t2, features, seq_idx))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return torch.cat(features, dim=2)
