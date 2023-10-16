#### TORCH Dataset Class 
import torch
import sys
sys.path.append(r'../../../')
from ssg2.data.rocksdbutils import RocksDBReader
import numpy as np
import cv2 




class RocksDBDataset_SSG2(torch.utils.data.Dataset):
    # This dataset produces both intersections and unions of labels 

    # I use CV2 for the distance transform
    # edt is good for multiclass distance transform.                                                                                    
    def __init__(self,          
                 flname_db,      
                 sequence_length, 
                 transform=None, 
                 lru_cache=0.01, # Relates to RocksDB cache memory size 
                 lru_cache_compr=0.01,   # Relates to RocksDB cache memory size 
                 num_workers = 4, 
                 read_only=True,
                 mode='train',
                 use_identity_input=False):  # You can include identical input in the sequence if you want, no performance difference in our experiments whether you use it or not.
        super().__init__()                                                                                                              
                                                                                                                                        
        self.mydbreader = RocksDBReader(flname_db,lru_cache,lru_cache_compr,num_workers,read_only)                                      
        self.length = len(self.mydbreader.keys) 

        self.transform = transform   
        if use_identity_input:
            assert sequence_length >=2
            self.sequence_length = sequence_length - 1 # first element is the same input
        else: 
            assert sequence_length >=1
            self.sequence_length = sequence_length  # first element is the same input

        self.use_identity_input = use_identity_input


        self.length_wth_labels =  self.sequence_length 
        self.mode = mode
        assert mode in ['train','valid'], ValueError("mode can be only train or valid, aborting...")

    def _add_null_class_to_segm_label(self,tlabels):
        # 1. Find common OFF class, that is where eveywhere is null                                                                     
        off_class = np.prod(1-tlabels,axis=0).astype(np.uint8)    
        tlabels = np.concatenate([tlabels,off_class[None]],axis=0).astype(np.uint8)
    
        return tlabels


    def _add_null_class_to_mtsk_label(self,tlabels):
        NClasses = tlabels.shape[0]//3 
        tlabels = self._add_null_class_to_segm_label(tlabels[:NClasses])
        # recalculate bounds and distance
        tbounds = self.get_boundary(tlabels)     
        tdist = self.get_distance(tlabels) # using cv2                                                                                  
        # Combine all in one label   
        tlabels = np.concatenate([tlabels,tbounds,tdist],axis=0) # cv2    

        #print (tlabels.shape)
        return tlabels.astype(np.float32)



    def get_boundary(self, _label, _kernel_size = (3,3)):               
                                                                                                                                        
        label = _label.copy().astype(np.uint8) # This is important otherwise it overwrites the original file                           
        for channel in range(label.shape[0]): 
            temp = cv2.Canny(label[channel],0,1) 
            label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)                  
        
        label  = label.astype(np.float32)
        label /= 255.# Scale to 0,1 
        label  = label.astype(np.uint8)                                                                                                  
        return label                                                                                                                    
                                                                                                                                        
    def get_distance(self,_label):     
        label = _label.copy()    
        dists = np.empty_like(label,dtype=np.float32)     
        for channel in range(label.shape[0]):            
            dist = cv2.distanceTransform(label[channel], cv2.DIST_L1, cv2.DIST_MASK_3) 
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)                 
            dists[channel] = dist                                                    
                                                                                    
        return dists                                                                                                                    
                                                                                                                                        
    def intersection(self,l1,l2):                                                                                                       
        l1 = l1.copy().astype(np.uint8)
        l2 = l2.copy().astype(np.uint8)                                                                                                        
        NClasses = l1.shape[0]//3 # segm/bounds/distance
        tt1   = l1[:NClasses]           
        tt2   = l2[:NClasses]                                                           
                                                                                         
        # The boundaries are dilated (thickened) so this is the way to get thick boundaries  
        # Intersection labels                                    
        tlabels = tt1 * tt2 # Set Intersection, can be done with min operation too 

        # @@@@@@@ NEW @@@@@@@@@@@@@@@@@                              
        # 1. Find common OFF class, that is where eveywhere is null   
        tlabels = self._add_null_class_to_segm_label(tlabels) 

        tbounds = self.get_boundary(tlabels)     
        tdist = self.get_distance(tlabels) # using cv2                                                                                  
                                               
        # Combine all in one label             
        tlabels = np.concatenate([tlabels,tbounds,tdist],axis=0) # cv2                                                                  

        #print (tlabels.shape)
        return tlabels.astype(np.float32)                                                                                               


    def set_union(self,l1,l2):
        l1 = l1.copy().astype(np.uint8)
        l2 = l2.copy().astype(np.uint8)                                                                                                        
        #print(l1.shape)                                                                                                    
        NClasses = l1.shape[0]//3 # segm/bounds/distance 
        tt1   = l1[:NClasses]                                                                                                           
        #tt1_b = l1[NClasses:2*NClasses]                
        tt2   = l2[:NClasses]
        tlabels = np.maximum(tt1,tt2)

        tlabels = self._add_null_class_to_segm_label(tlabels)

        tbounds = self.get_boundary(tlabels) 
        tdist = self.get_distance(tlabels) # using cv2                                                                                  

        # Combine all in one label          
        tlabels = np.concatenate([tlabels,tbounds,tdist],axis=0) # cv2                                                                  

        #print (tlabels.shape)
        return tlabels.astype(np.float32)                                                                                               


    def set_diff(self,l1,l2):
        # First find intersection 
        l1 = l1.copy().astype(np.uint8)
        l2 = l2.copy().astype(np.uint8)                                                                                                        
        NClasses = l1.shape[0]//3 # segm/bounds/distance                                                                                
        tt1   = l1[:NClasses]                                                                                                           
        tt2   = l2[:NClasses]
        tlabels = np.minimum(tt1,tt2)

        # Now get difference 
        tlabels = tt1 - tlabels 
        
        # Fix it 
        tlabels = self._add_null_class_to_segm_label(tlabels)

        tbounds = self.get_boundary(tlabels) 
        tdist = self.get_distance(tlabels) # using cv2                                                                                  

        # Combine all in one label          
        tlabels = np.concatenate([tlabels,tbounds,tdist],axis=0) # cv2                                                                  

        return tlabels.astype(np.float32)                                                                                               

                                                                                                                                        
    def __len__(self): 
        return self.length     

    def __getitem__(self,idx):
        inputs1, labels1 = self.mydbreader.get_inputs_labels(idx)
        labels1 = labels1.astype(np.float32)
        NClasses = labels1.shape[0]//3 
        labels1[2*NClasses:] = labels1[2*NClasses:] / 255. # SCALE DOWN THE DISTANCE TRANSFORM
        NClasses_init = labels1.shape[0]//3                     
        if self.transform is not None:                         
            inputs1, labels1   = inputs1.astype(np.float32),  labels1.astype(np.float32)
            inputs1, labels1   = self.transform(inputs1, labels1)
        

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ SET INTERSECTION LABELS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Secondary inputs - Sequence
        lstm_seq_in = []

        lstm_seq_inter_gt = []  
        lstm_seq_union_gt = []  
        lstm_seq_diffs_gt = []  
        
        # -------------------------------------------------------------------------------------------------------------
        

     
        labels1_wth_off = self._add_null_class_to_mtsk_label(labels1) # labels 

        #labels2_seq = []
        for _ in range(self.length_wth_labels): 
            idx_rand           = np.random.randint(self.length) # Select a random index from the allowed labelled dataset 
            inputs2, labels2   = self.mydbreader.get_inputs_labels(idx_rand)
            labels2 = labels2.astype(np.float32)
            labels2[2*NClasses:] = labels2[2*NClasses:] / 255. # SCALE DOWN THE DISTANCE TRANSFORM

            if self.transform is not None:  
                inputs2,  labels2 = inputs2.astype(np.float32), labels2.astype(np.float32)  
                inputs2, labels2 = self.transform(inputs2, labels2)  
            
            labels_set_iner = self.intersection(labels1,labels2)   
            labels_set_union = self.set_union(labels1,labels2)   
            labels_set_diffs  = self.set_diff(labels1,labels2)   


            lstm_seq_in.append(inputs2[None])   
            lstm_seq_inter_gt.append(labels_set_iner[None])
            lstm_seq_union_gt.append(labels_set_union[None])
            lstm_seq_diffs_gt.append(labels_set_diffs[None])


        if self.use_identity_input:
            # Add identity input
            lstm_seq_in.append( inputs1[None] )# image 
            lstm_seq_inter_gt.append( labels1_wth_off[None] )
            lstm_seq_union_gt.append( labels1_wth_off[None] )
            lstm_seq_diffs_gt.append( np.zeros_like(labels1_wth_off[None]) )


        # -------------------------------------------------------------------------------------------------------------
        inputs2              = np.concatenate(lstm_seq_in,axis=0)   # Sequence inputs 
        labels_intersection  = np.concatenate(lstm_seq_inter_gt,axis=0)   # Set Intersection labels 
        labels_union         = np.concatenate(lstm_seq_union_gt,axis=0)   # Set Union labels 
        labels_diffs         = np.concatenate(lstm_seq_diffs_gt,axis=0)   # Set diff labels 
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # Bring Sequence in dimension 1 
        inputs2 		= inputs2.transpose([1,0,2,3])
        labels_intersection 	= labels_intersection.transpose([1,0,2,3])  
        labels_union		= labels_union.transpose([1,0,2,3]) 
        labels_diffs		= labels_diffs.transpose([1,0,2,3])


        # Random shuffle elements in sequence but PRESERVE the corresponding image / label locations within sequence.
        c = np.arange(labels_intersection.shape[1])
        np.random.shuffle(c)



        # inputs1:  channels x height x width
        # inputs
        # innputs2  sequence x channels x height x width 
        return inputs1, inputs2[:,c], labels_intersection[:,c],  labels_union[:,c], labels_diffs[:,c],  labels1_wth_off








