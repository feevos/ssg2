import sys
sys.path.append(r'../../../') # Location of ssg2 repository 
from ssg2.data.isprs.DatasetCreation_defs import *
from ssg2.data.rocksdbutils_v2 import Rasters2RocksDB

if __name__=='__main__':
 
    
    tuples = ISPRS_TuplesCreationIterator(data_dir=r'./DataRaw/') # Change data_dir with where location of raw data is 

    
    isprs_creation_transform = DatasetCreationTransform(NClasses=NClasses, 
            creation_transform=ISPRSNormal(),
            representation=representation_1h)

    # Save database in location 
    flname_prefix_save = './TRAINING_DBs/F256/'
    # Create directory it if it doesn't exist 
    os.makedirs(flname_prefix_save, exist_ok=True)

    myr2db = Rasters2RocksDB(
        tuples,ISPRS_names2raster_tuple,
        # ############ CHANGE HERE IF YOU WANT F128 ######
        metadata=meta_1hot_f256, # <--- 
        Filter=F256, # < ------ 
        # ###############################################
        flname_prefix_save = flname_prefix_save,
        transform=isprs_creation_transform)

    # Create RocksDB Database to store the data 
    myr2db.create_dataset()
    
    
