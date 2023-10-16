import rasterio 
from math import ceil as mceil
import numpy as np 
import os 


class _RasterIterableBase(object):
    # An iterable version of raster sliding window patches 
    # Shape assumes Channels x Height x Width format 

    def __init__(self, shape, Filter=256, stride_divisor=2, batch_size=None):
        super().__init__()

        self.shape = shape
        self.F = Filter
        self.s = Filter//stride_divisor
        self.generate_slices()
        if batch_size is not None:
            self.batch_size = batch_size
            self.batchify(batch_size)

    def get_len_set(self):
        return len(self.RowsCols)

    def get_len_batch_set(self):
        return len(self.BatchRowsCols)

    def generate_slices(self):

        shape = self.shape 
        # Constants that relate to rows, columns 
        self.nTimesRows = int((shape[-2] - self.F)//self.s + 1)
        self.nTimesCols = int((shape[-1] - self.F)//self.s + 1)


        # Use these directly 
        RowsCols = [(row, col) for row in range(self.nTimesRows-1) for col in range(self.nTimesCols-1)]
        RowsCols_Slices = [ (slice(row*self.s,row*self.s +self.F,1),slice(col*self.s,col*self.s+self.F,1) )  for (row,col) in RowsCols ]

        # Construct RowsCols for last Col 
        col_rev = shape[-1]-self.F
        Rows4LastCol = [(row,col_rev) for row in range(self.nTimesRows-1)]
        Rows4LastCol_Slices = [ (slice(row*self.s,row*self.s +self.F,1),slice(col_rev,col_rev+self.F,1) )  for (row,col_rev) in Rows4LastCol]

        # Construct RowsCols for last Row 
        row_rev = shape[-2]-self.F
        Cols4LastRow        = [(row_rev,col) for col in range(self.nTimesCols-1)]
        Cols4LastRow_Slices = [(slice(row_rev,row_rev+self.F,1),slice(col*self.s,col*self.s +self.F,1) )  for (row_rev,col) in Cols4LastRow]

        
        # Store all Rows and Columns that correspond to raster slices and slices 
        self.RowsCols           = RowsCols + Rows4LastCol + Cols4LastRow
        self.RowsCols_Slices    = RowsCols_Slices + Rows4LastCol_Slices + Cols4LastRow_Slices


    def batchify(self,batch_size):
        n = mceil(len(self.RowsCols)/batch_size)
        self.BatchIndices  = np.array_split(list(range(len(self.RowsCols))),n,axis=0)
        self.BatchRowsCols = np.array_split(self.RowsCols,n,axis=0)
        self.BatchRowsCols_Slices = np.array_split(self.RowsCols_Slices,n,axis=0)






from pathos.pools import ThreadPool as pp
class RasterMaskIterableInMemory(_RasterIterableBase):
    # This will accept read in windows from Rasterio 

    def __init__(self, lst_of_rasters, Filter=256, stride_divisor=2, transform=None,  batch_size=None, num_workers=28, 
            batch_dimension=False):
        self.lst_of_rasters = lst_of_rasters
        assert len(lst_of_rasters) >= 2, ValueError("You need at least two files, an input image and a target mask, you provided::{}".format(self.number_of_rasters))
        shape = lst_of_rasters[0].shape

        for idx in range(1,len(lst_of_rasters)):
            assert shape[-2:] == lst_of_rasters[idx].shape[-2:], ValueError("All rasters in the list must have the same spatial dimensionality ")

        super().__init__(shape=shape, Filter=Filter, stride_divisor=stride_divisor, batch_size=batch_size)
        self.transform   = transform
        self.num_workers = num_workers

        self.batch_dimension = batch_dimension


    def get_element(self,idx):
        slice_row, slice_col = self.RowsCols_Slices[idx] 
        patches = []
        for raster in self.lst_of_rasters:
            patches.append(raster[...,slice_row,slice_col])
            
        if self.transform is not None:
            patches = self.transform(*patches)
            
        return patches


    def get_batch(self, idx):
        batch_patches = []
        for slice_row,slice_col in self.BatchRowsCols_Slices[idx]: # Batch Indices 
            patches = []
            for raster in self.lst_of_rasters:
                patches.append(raster[...,slice_row,slice_col][None]) # Add batch dimension
                #patches.append(raster[...,slice_row,slice_col])
            batch_patches.append(patches)

        if self.transform is not None:
            # Trick to go from list to arguments
            def vtransform(patch):
                tpatch = [p[0] for p in patch] # Remove batch dim for transform
                tpatch = self.transform(*tpatch)
                #tpatch = [p[None] for p in tpatch] # Restore batch dim
                return tpatch
        
            pool = pp(nodes=self.num_workers)
            result = pool.map(vtransform,batch_patches)
            batch_patches = result
        if not self.batch_dimension:
            return batch_patches
        # Now concatenate all along the first dimension?
        lst_of_elements_in_patch = zip(*batch_patches)
        batched_elements = []
        for tinput in lst_of_elements_in_patch:
            tinput = [t[None] for t in tinput] # Add batch dimension
            batched_elements.append(np.concatenate(tinput,axis=0))


        return batched_elements

