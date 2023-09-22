import torch
from ssg2.nn.layers.conv2Dnormed import *




class UpSample2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest', norm_type = 'BatchNorm', norm_groups=None):
        super(UpSample2D,self).__init__()

        # Changes spatial size 
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)
        # Fixes number of channels 
        self.convup_normed= Conv2DNormed(in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3,
                padding=1,
                norm_type=norm_type,
                num_groups=norm_groups)


    def forward(self,input):
        out = self.upsample(input)
        out = self.convup_normed(out)

        return out 




