import numpy as np

class ISPRSNormal(object):
    """
    class for Normalization of images, per channel, in format CHW 
    """
    def __init__(self):

        # The DEMS is calculated from the 1_DEMS.rar archive 
        self._mean = np.array([85.846375, 91.77658 , 84.99307 , 97.275826, 38.462185]).astype(np.float32)
        self._std = np.array ([35.466473 , 34.850277 , 36.257114 , 35.685856 ,  6.2159314] ).astype(np.float32)


    def __call__(self,img):

        temp = img.astype(np.float32)
        temp2 = temp.T            
        temp2 -= self._mean
        temp2 /= self._std
            
        temp = temp2.T

        return temp
        

    def restore(self,normed_img):
        d2 = normed_img.T * self._std
        d2 = d2 + self._mean
        d2 = d2.T
        d2 = np.round(d2)
        d2 = d2.astype('uint8')

        return d2 



