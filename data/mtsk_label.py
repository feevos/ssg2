import cv2
import numpy as np
class MultiTaskingLabelsCreationTransform(object):
    # This is a convenience class for creating segmentation/boundaries and distance transform for the case of multiclass segmentation problems.
    def __init__(self, NClasses, compress=255.):
        self.NClasses = NClasses
        self.compress = compress # Level of compression for uint16 representation of labels 

        self.representation_1h = np.eye(NClasses)

    def get_boundary(self, _label, _kernel_size = (3,3)):
        
        label = _label.copy() # This is important otherwise it overwrites the original file 
        for channel in range(label.shape[0]):
            temp = cv2.Canny(label[channel],0,1)
            label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)

        label = label.astype(np.float32)
        label /= self.compress # Scales to 0,1
        label = label.astype(np.uint8)

        return label

    def get_distance(self,_label):

        label = _label.copy()
        dists = np.empty_like(label,dtype=np.float32)
        for channel in range(label.shape[0]):
            dist = cv2.distanceTransform(label[channel], cv2.DIST_L1, cv2.DIST_MASK_3)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dists[channel] = dist

        # Compress only on 1Hot representation
        dists = dists * self.compress # compress=255 
        dists = dists.astype(np.uint8)
        return dists


    def __call__(self,label):

        # Squash dimension 1
        label = np.squeeze(label)
        if len(label.shape)==2:
            label1h = self.representation_1h[label].transpose([2,0,1]).astype(np.uint8)
        else:
            label1h = label

        label1h = label1h.astype(np.uint8)
        bounds = self.get_boundary(label1h)
        dist   = self.get_distance(label1h)


        label = np.concatenate([label1h,bounds,dist],axis=0).astype(np.uint8)

        return label

