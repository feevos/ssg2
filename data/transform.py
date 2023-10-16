from .mtsk_label import *  
class DatasetCreationTransform(object):
    def __init__(self,NClasses, creation_transform, compress=255.):
        self.inputs_trans = creation_transform # Standardization
        self.labels_trans = MultiTaskingLabelsCreationTransform(NClasses=NClasses, 
                                                                compress=compress)

        self.compress = compress

    def __call__(self,inputs, labels):
        labels = labels #/self.compress
        labels = labels.astype(np.uint8)
        inputs = self.inputs_trans(inputs) # np.float32
        labels = self.labels_trans(labels) # np.uint8 

        return inputs, labels




import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform
class RSRandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.
    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        norm,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ):

        self.trans = A.RandomBrightnessContrast(brightness_limit=brightness_limit,contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, always_apply=always_apply, p=p)
        self.norm = norm
    

    def __call__(self,timg):
        timg_denormed = self.norm.restore(timg)
        tscale = np.iinfo(str(timg_denormed.dtype)).max
        timg_denormed = timg_denormed.astype(np.float32) / tscale
        
        transformed = self.trans(image=timg_denormed.transpose([1,2,0]))['image']
        transformed = transformed * tscale
        transformed = self.norm(transformed.transpose([2,0,1]))
        return transformed
        
        
       
class TrainingTransform(object):
    # Built on Albumentations, this provides geometric transformation only
    def __init__(self, NClasses, prob = 1., mode='train', compress=255., norm = None):

        self.NClasses = NClasses
        self.distance_scale=1. / compress # This is the scaling of the mask distance transform

        self.geom_trans = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.ElasticTransform(p=1), 
                        A.GridDistortion(distort_limit=0.4,p=1.),
                        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), 
                        ],p=1.)
                    ],
            p = prob)

        self.norm = norm
        if norm is not None:
            self.img_trans = RSRandomBrightnessContrast(norm = norm,brightness_limit=.1,contrast_limit=.1, brightness_by_max=False, p=1) 
            
            
        if mode=='train':
            self.mytransform = self.transform_train
        elif mode =='valid':
            self.mytransform = self.transform_valid
        else:
            raise ValueError('transform mode can only be train or valid')

    def transform_valid(self, timg, tmask):
        tmask = tmask.copy()
        tmask[2* self.NClasses:] = tmask[ 2* self.NClasses:] * self.distance_scale

        return timg, tmask

    def transform_train(self, timg, tmask):
        tmask = tmask.copy()
        tmask[2* self.NClasses:] = tmask[2* self.NClasses:] * self.distance_scale


        if isinstance(timg,list):
            nitems = len(timg)
            if self.norm is not None:
                timg = [self.img_trans(ttimg) for ttimg in timg] 
            timg = np.concatenate(timg,axis=0)

            # Special treatment of time series
            if len(timg.shape) == 4:
                c,t,h,w = timg.shape
                timg = timg.reshape(c*t,h,w)
                result = self.geom_trans(image=timg.transpose([1,2,0]),mask=tmask.transpose([1,2,0]))
                timg_t, tmask_t = result.values()
                timg_t = timg_t.transpose([2,0,1])
                tmask_t = tmask_t.transpose([2,0,1])

                timg_t = timg_t.reshape(c,t,h,w)

            else:
                result = self.geom_trans(image=timg.transpose([1,2,0]),mask=tmask.transpose([1,2,0]))
                timg_t, tmask_t = result.values()
                timg_t = timg_t.transpose([2,0,1])
                tmask_t = tmask_t.transpose([2,0,1])
            
            # Restore original list 
            timg_t = np.split(timg_t,nitems,axis=0)

        else:
            if self.norm is not None:
                timg = self.img_trans(timg) 
 
            # Special treatment of time series
            if len(timg.shape) == 4:
                c,t,h,w = timg.shape
                timg = timg.reshape(c*t,h,w)
                result = self.geom_trans(image=timg.transpose([1,2,0]),mask=tmask.transpose([1,2,0]))
                timg_t, tmask_t = result.values()
                timg_t = timg_t.transpose([2,0,1])
                tmask_t = tmask_t.transpose([2,0,1])

                timg_t = timg_t.reshape(c,t,h,w)

            else:
                result = self.geom_trans(image=timg.transpose([1,2,0]),mask=tmask.transpose([1,2,0]))
                timg_t, tmask_t = result.values()
                timg_t = timg_t.transpose([2,0,1])
                tmask_t = tmask_t.transpose([2,0,1])

        return timg_t, tmask_t

    def __call__(self, timg, tmask):        
        return self.mytransform(timg,tmask) 










