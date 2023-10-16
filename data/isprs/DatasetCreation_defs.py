import rasterio 

from ssg2.data.rasteriter import RasterMaskIterableInMemory
from ssg2.data.transform import DatasetCreationTransform
from ssg2.utils.progressbar import progressbar

import sys
sys.path.append(r'../../..') # Depends on the source code - should point on ssg2 repository 
from ssg2.data.isprs.normal import ISPRSNormal


import numpy as np
import glob
import os 

# These are our classes in RGB format, there exist also the 
# corresponding integer values commented out.
# ******************************************************************
Background = np.array([255,0,0]) #:{'name':'Background','cType':0},
ImSurf = np.array ([255,255,255])# :{'name':'ImSurf','cType':1},
Car = np.array([255,255,0]) # :{'name':'Car','cType':2},
Building = np.array([0,0,255]) #:{'name':'Building','cType':3},
LowVeg = np.array([0,255,255]) # :{'name':'LowVeg','cType':4},
Tree = np.array([0,255,0]) # :{'name':'Tree','cType':5}
# ******************************************************************


def rgb_to_2D_label(_label):
    """
    Here _label is the mask raster that corresponds to the input image.
    """
    label_seg = np.zeros(_label.shape[1:],dtype=np.uint8)
    label_seg [np.all(_label.transpose([1,2,0])==Background,axis=-1)] = 0
    label_seg [np.all(_label.transpose([1,2,0])==ImSurf,axis=-1)] = 1
    label_seg [np.all(_label.transpose([1,2,0])==Car,axis=-1)] = 2
    label_seg [np.all(_label.transpose([1,2,0])==Building,axis=-1)] = 3
    label_seg [np.all(_label.transpose([1,2,0])==LowVeg,axis=-1)] = 4
    label_seg [np.all(_label.transpose([1,2,0])==Tree,axis=-1)] = 5
    
    return label_seg



# Helper functions to get the tuples of names 
import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_keys(some_filename):
    lst_of_keys = [ n.lstrip('0') for n in some_filename.split("/")[-1].split(".")[0].split("_")[2:]]
    return "_".join(lst_of_keys[:2])


def ISPRS_TuplesCreationIterator(data_dir='./DataRaw/', use_subset=False):

    flnames_labels = sorted(glob.glob(os.path.join(data_dir,'5_Labels_for_participants','*.tif')),key=natural_key)
    labels_keys    = [get_keys(name) for name in flnames_labels]

    
    flnames_dems   = sorted(glob.glob(os.path.join(data_dir,'1_DSM','*.tif')),key=natural_key)
    flnames_rgbir  = sorted(glob.glob(os.path.join(data_dir,'4_Ortho_RGBIR','*.tif')),key=natural_key)

    
    flnames_rgb_clean = []
    for key in labels_keys:
        flnames_rgb_clean.append(list(filter(lambda x: key in get_keys(x),flnames_rgbir))[0])
    
    flnames_dems_clean = []
    for key in labels_keys:
        flnames_dems_clean.append(list(filter(lambda x: key in get_keys(x),flnames_dems))[0])
    
    
    return list(zip(flnames_rgb_clean, flnames_dems_clean, flnames_labels))


def ISPRS_names2raster_tuple_combined_dems(names):
    rgbd, gt = names
    with rasterio.open(rgbd,mode='r',driver='GTiff') as src:
        rgbd = src.read()
    with rasterio.open(gt,mode='r',driver='GTiff') as src:
        gt = src.read()
    gt = rgb_to_2D_label(gt)

    
    lst_of_rasters = [rgbd,gt[None]]

    return lst_of_rasters


import cv2
def ISPRS_names2raster_tuple(names):
    rgb,dems,gt = names
    with rasterio.open(rgb,mode='r',driver='GTiff') as src:
        rgb = src.read()
    with rasterio.open(dems,mode='r',driver='GTiff') as src:
        dems = src.read()
    with rasterio.open(gt,mode='r',driver='GTiff') as src:
        gt = src.read()
    gt = rgb_to_2D_label(gt)

    # fixes bug  in data
    if (dems.shape[1:] != rgb.shape[1:]):
        # resize image
        dems = cv2.resize(dems.transpose([1,2,0]), rgb.shape[1:], interpolation = cv2.INTER_AREA)
        dems = dems[None] # dems has single channel 

    rgbd = np.concatenate([rgb,dems],axis=0)
    
    lst_of_rasters = [rgbd,gt[None]]

    return lst_of_rasters


# CONSTANTS 
NClasses= 6
F256 = 256
from collections import OrderedDict as odict 
representation_1h = '1Hot'
meta_1hot_f256 = odict({'inputs'  : odict({'inputs_shape' : (5, F256, F256), 
                                    'inputs_dtype' : np.float32}),
                 'labels'  : odict({'labels_shape' : (3*NClasses, F256, F256),
                                    'labels_dtype' : np.uint8})})

F128=128
meta_1hot_f128 = odict({'inputs'  : odict({'inputs_shape' : (5, F128, F128), 
                                    'inputs_dtype' : np.float32}),
                 'labels'  : odict({'labels_shape' : (3*NClasses, F128, F128),
                                    'labels_dtype' : np.uint8})})
