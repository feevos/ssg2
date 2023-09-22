import torch
from ssg2.models.head.head_cmtsk import * 

from ssg2.models.ssg2.ptavit3d   import MaxVitStage3D_no_down

from trchprosthesis.nn.layers.scale import *
from trchprosthesis.experimental.nn.layers.patchattention import  PatchAttention2D
from trchprosthesis.nn.activations.sigmoid_crisp import *
from trchprosthesis.nn.layers.conv2Dnormed import *



__all__ = ['head_cmtsk_3D']





# Need correlation in time, so need custom attention that does Time x Time comparison 
#from trchprosthesis.experimental.nn.units.maxvit_fd import MaxVitStage_no_down

# In this version I add attention on the total of least of features, and then I consume them with head_lstm 
from ptavit3d import PTAViTStage3D_no_down
class head_cmtsk_3D(torch.nn.Module):
    def __init__(self, nfilters, NClasses, spatial_size=256, norm_type = 'GroupNorm', norm_groups=4, segm_act ='sigmoid', nresblocks=2):
        super().__init__()
        # TODO: Not sure if I should detach features before inserting to the lstm, in order to train separately or not
        # Different heads decouple the features interpetation, the ones created by the convlstmcell
        # this was found empirically, when using shared head for both lstm features and base features, performance is noise and overfitting occurs. 

        self.NClasses = NClasses

        self.head_inters = head_cmtsk(nfilters, NClasses, norm_type, norm_groups,representation, segm_act)
        self.head_unions = head_cmtsk(nfilters, NClasses, norm_type, norm_groups,representation, segm_act)
        self.head_diffs  = head_cmtsk(nfilters, NClasses, norm_type, norm_groups,representation, segm_act)

        self.compressor   = torch.nn.Conv2d((nfilters+2*3*NClasses),nfilters,kernel_size=1,bias=False)
        self.head_target = head_cmtsk(nfilters, NClasses, norm_type, norm_groups,representation, segm_act)

        # This is not a nice solution, because I need cross spatial attention. The FracTAL cannot do cross spatial, but is good for prototyping
        scales = 16//(256//spatial_size)
        scales = (scales,scales) # These are spatial only 
        
        self.conv_sequence = MaxVitStage3D_no_down(
                        layer_dim_in=nfilters,
                        layer_dim=nfilters,
                        layer_depth = nresblocks,
                        nheads=nfilters//4,
                        scales=scales)

    #   Fuzzy set intersection 
    def fz_tnorm(self, x,y,p=1.e-5):
        tprod = x*y
        denum = p + (1.-p)*(x+y-tprod)
        return tprod/denum

    # Fuzzy set union 
    def fz_conorm(self,x,y,p=1.e-5):
        return 1.-self.fz_tnorm(1-x,1-y,p)


    # During inference we can avoid calculating some layers, for faster and more memory efficient operation. 
    def forward_inference(self, lst_of_features):
        lst_of_features = lst_of_features.permute(0,2,1,3,4).contiguous()
        b,c,n,h,w = lst_of_features.shape
        
        outs_inter          = []
        outs_union          = []

        #for features in lst_of_features:
        for seq_idx in range(n):
            features = lst_of_features[:,:,seq_idx]
            #print (n, seq_idx, features.shape)
            preds_inter =  self.head_inters(features)
            outs_inter.append(preds_inter.unsqueeze(2)) # Intermediate outputs
            preds_union =  self.head_unions(features)
            outs_union.append(preds_union.unsqueeze(2))
            
            # inter \cup diff is target 

        outs_inter      = torch.cat(outs_inter,dim=2)
        outs_union      = torch.cat(outs_union,dim=2)


        # Each of the elements of the sequence must equal the target. 
        # HAS NCLASSES dimensionality, not multitasking

        uint = torch.max(outs_inter,dim=2)[0] # union of intersections PRIOR
        iuni = torch.min(outs_union,dim=2)[0] # intersection of unions PRIOR

        #lst_of_features = lst_of_features.reshape(b,c,n,h,w)
        #print (lst_of_features.shape)
        # This utilizes cross Sequence-Spatial correlation
        out3d2d = self.conv_sequence(lst_of_features)# Average across sequence dim 
        #print(out3d2d.shape)
        features_target =  self.compressor(torch.cat([out3d2d.mean(dim=2) ,uint,iuni],dim=1))

        preds_target = self.head_target(features_target)
        #return outs_inter, outs_union,  preds_target 
        return  preds_target
        



    def forward(self, lst_of_features):
        lst_of_features = lst_of_features.permute(0,2,1,3,4).contiguous()
        b,c,n,h,w = lst_of_features.shape
        
        outs_inter          = []
        outs_union          = []
        outs_diffs          = []

        outs_target_fz      = []
        outs_null_fz        = []

        #for features in lst_of_features:
        for seq_idx in range(n):
            features = lst_of_features[:,:,seq_idx]
            #print (n, seq_idx, features.shape)
            preds_inter =  self.head_inters(features)
            outs_inter.append(preds_inter.unsqueeze(2)) # Intermediate outputs
            preds_union =  self.head_unions(features)
            outs_union.append(preds_union.unsqueeze(2))
            preds_diff = self.head_diffs(features)
            outs_diffs.append(preds_diff.unsqueeze(2))
            
            # inter \cup diff is target 
            pred_target_fz = self.fz_conorm(preds_inter[:,:self.NClasses],preds_diff[:,:self.NClasses]) # This is very good 
            outs_target_fz.append( pred_target_fz.unsqueeze(2))


            # Intersection of intersection with diff is null
            pred_null_fz = self.fz_tnorm(preds_inter[:,:self.NClasses],preds_diff[:,:self.NClasses])
            outs_null_fz.append( pred_null_fz.unsqueeze(2) )

        outs_inter      = torch.cat(outs_inter,dim=2)
        outs_union      = torch.cat(outs_union,dim=2)
        outs_diffs      = torch.cat(outs_diffs,dim=2)


        # Each of the elements of the sequence must equal the target. 
        # HAS NCLASSES dimensionality, not multitasking
        outs_target_fz   = torch.cat(outs_target_fz,dim=2) # maps to TARGET
        outs_null_fz     = torch.cat(outs_null_fz,dim=2) # maps to NULL (zeros) 

        uint = torch.max(outs_inter,dim=2)[0] # union of intersections PRIOR
        iuni = torch.min(outs_union,dim=2)[0] # intersection of unions PRIOR

        #lst_of_features = lst_of_features.reshape(b,c,n,h,w)
        #print (lst_of_features.shape)
        # This utilizes cross Sequence-Spatial correlation
        out3d2d = self.conv_sequence(lst_of_features)# Average across sequence dim 
        #print(out3d2d.shape)
        features_target =  self.compressor(torch.cat([out3d2d.mean(dim=2) ,uint,iuni],dim=1))


        preds_target = self.head_target(features_target)
        #return outs_inter, outs_union,  preds_target 
        return outs_inter, outs_union, outs_diffs, preds_target, outs_target_fz,  outs_null_fz, out3d2d
        



