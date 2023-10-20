# RocksDB-based Custom Dataset for Deep Learning in Remote Sensing Applications 


## Introduction 

In the field of Remote Sensing, we deal with large volumes of data. These are usually large raster files (that do not fit into RAM memory) that we need to split into smaller training chips and store into some format. Usually, due to RAM limitations, we are forced to read iteratively from hard drive during training, and that induces one of the main reasons for bad deep learning training performance: I/O bottleneck. Therefore, how we choose to save these training chips, can make or break our training. The easiest approach, that is followed widely, is to save numpy arrays in numpy  (compressed or native) format directly to the hard drive. This results in a large number of files on disk, which is not always optimal. Here are some of the key challenges we face based on the available options:

1. **File Size and Speed**: Uncompressed files are not only storage-heavy but also slow to read. While compressing these files could alleviate storage concerns and improve read speed, it brings along the overhead of CPU-intensive decompression during the training process. This depends on the compression format and the algorithmic implementations. 
2. **The Delicate Balancing Act**: Finding the sweet spot between file size and CPU consumption is a non-trivial task. This becomes particularly relevant when dealing with datasets so large that they don't fit into memory, necessitating on-the-fly reading and decompression.
3. **Partial Caching**: Solutions like HDF5 offer limited flexibility when it comes to partial caching, essentially forcing us into an all-or-none scenario. This is far from ideal when we have finite resources but would like to cache a subset of the data for faster access.
4. **Data Retrieval**: In a stochastic training environment, where data points are often selected randomly, the efficiency of the data retrieval system can add an additional performance penalty. The ability to quickly find and load a specific datum is crucial for maintaining a smooth training process.

To tackle these challenges, the RocksDB-based dataset class provided in this repository offers a robust and efficient mechanism for data storage and retrieval. Here's how RocksDB comes into play:

1. **Compression Algorithms**: RocksDB provides an array of compression algorithms, making it versatile for a variety of applications. You can customize the algorithm according to your needs, but the default options are usually sufficient for most use-cases. This feature helps us find the optimal balance between file size and read speed, mitigating the trade-offs involved.
2. **Bloom Filters for Efficient Searching**: One of the standout features of RocksDB is its use of [Bloom filters](https://github.com/facebook/rocksdb/wiki/RocksDB-Bloom-Filter) for rapid key lookups. This is particularly important when your dataset has tens of thousands of entries or more. The efficiency gained through this feature significantly speeds up the data retrieval process, thereby minimizing bottlenecks during training. 
3. **Partial Data Caching**: RocksDB doesn't enforce an all-or-nothing approach when it comes to data caching. Instead, it allows for partial caching based on a user-defined size limit (in GB). This means that even if your entire dataset can't fit into memory, RocksDB will cache as much as possible within the provided size limit, enabling faster data access without overwhelming your system resources. 


## Understanding how RocksDB works (python interface - BASICS). 


For this project we will use the python implementation of RocksDB as provided in [lbry-rocksdb](https://github.com/lbryio/lbry-rocksdb) (see also [NightTsarina](https://github.com/NightTsarina/python-rocksdb) implementation. For documentation users will find [this](https://rocksdb-tina.readthedocs.io/en/latest/) resource very helpful. Installation of lbry-rocksdb exists in the provided containers. 

OK, lets dive in, this is how one stores key (here ```b"somekey"```) value (here: ```b"SomeValue"```) in rocksdb interface:

```Python    
import rocksdb     
db = rocksdb.DB("YourDataBaseName.db", rocksdb.Options(create_if_missing=True)) # Creates the Database     
db.put(b"somekey", b"SomeValue") # Store key-value pair 
print (db.get(b"somekey")) # Retrieve store value     
db.close() # Close database
```       

If we look into the directory where the above database was opened, we will see something like: 

```shell
computername:/workdir$tree
.
└── YourDataBaseName.db
    ├── 000005.log
    ├── CURRENT
    ├── IDENTITY
    ├── LOCK
    ├── LOG
    ├── MANIFEST-000004
    └── OPTIONS-000007

```
These are the files created by the RocksDB software, and we don't really care too much about whats in there (I don't!). Now, the previous example was based on binary strings (both for key as well as value pairs). So how do we store numpy arrays - which is the format we care about when creating a dataset class for pytorch (or other DL framework). The answer is: we need to translate them to raw bytes before storage. Now, I know that a lot of options exist out there (pickle) for this, however none is as fast as native numpy operations. For simplicity, lets assume that we want to save a single numpy tensor, of shape (3,128,128). This can be achieved with the following: 

```python
key=b'image_1'
timg = np.random.rand(3,128,128).astype(np.float32)
valueinbytes = timg.tobytes()
db.put(key,valueinbytes)
```

we retrieve these values in numpy format with 

```python
retrieved_valueinbytes = db.get(key)
retrieved_numpy = np.frombuffer(retrieved_valueinbytes,dtype=np.float32).reshape(3,128,128)) 
print( (retrieved_numpy==timg).all() ) # True
```

Importantly, we need to provide the original shape and datatype of the retrieved numpy array before translating it to original numpy format. Now, this burden can be aleviated if we were to use pickle.dumps/loads, but that would be much slower (to the point of diminishing gains). Therefore it is not recommended. Of course it is possible to store key value pairs in a similar way: 

```python
# Assuming same shape/dtype for all data in database
img_shape   = (3,128,128)
img_dtype   = np.float32
label_shape = (1,128,128)
label_dtype = np.uint8

# datum_index=123
key_img   = b'image_123'
key_label = b'label_123'
timg      = np.random.rand(*img_shape).astype(img_dtype)
tlabel    = np.random.rand(*label_shape).astype(label_dtype) # 1hot encoding

# Store values into database 
db.put(key_img,timg.tobytes())
db.put(key_label,tlabel.tobytes())

# Retrieved values 
timg_retrieved    = np.frombuffer(db.get(key_img),dtype=img_dtype).reshape(*img_shape)
tlabel_retrieved = np.frombuffer(db.get(key_label),dtype=label_dtype).reshape(*label_shape)

(timg_retrieved == timg).all() # True
(tlabel_retrieved == tlabel).all() # True
```

One more comment here to be added is that storing as independent images and labels may increase the time of search, so it is better to store them in column-families (or else: binding them together), this example is self explanatory (I hope - Note, you may want to delete the previous database you created, i.e. the whole directory, before proceeding): 

```python
import rocksdb
import numpy as np

opts = rocksdb.Options(create_if_missing=True)
db = rocksdb.DB("YourDatabase.db",opts,read_only=False) 


# Create Column families for inputs and labels 
# Inputs
opts_inputs = rocksdb.ColumnFamilyOptions()
db.create_column_family(b'inputs',opts_inputs)

# labels
opts_labels = rocksdb.ColumnFamilyOptions()
db.create_column_family(b'labels',opts_labels)

# Now the standard machinery applies for storing and retrieving keys 
img_shape   = (3,128,128)
img_dtype   = np.float32
label_shape = (1,128,128)
label_dtype = np.uint8

# datum_index=123
# But we only need one key as we are storing images and labels in different column families
key   = b'123'
timg      = np.random.rand(*img_shape).astype(img_dtype)
tlabel    = np.random.rand(*label_shape).astype(label_dtype) # 1hot encoding


cf_in = db.get_column_family(b'inputs')
db.put((cf_in,key),timg.tobytes())

cf_label = db.get_column_family(b'labels')
db.put((cf_label,key),tlabel.tobytes())


# Now read back: 
cf_in    = db.get_column_family(b'inputs')
cf_labels = db.get_column_family(b'labels')
timg_retrieved   = np.frombuffer(db.get((cf_in,key)),dtype=img_dtype).reshape(*img_shape)
tlabel_retrieved = np.frombuffer(db.get((cf_labels,key)),dtype=label_dtype).reshape(*label_shape)

```

That's it, that is the basis that will help you understand the class definitions under [ssg2/data/rocksdbutils.py](ssg2/data/rocksdbutils.py) - see also [ssg2/Notebooks/Demo_RocksDB_tutorial.py](ssg2/Notebooks/Demo_RocksDB_tutorial.py). To give some numbers perspective, for the mnist dataset we found Writing speedup x6 in comparison with standard HDF5 (DELL Laptop Precision with 11th Gen Intel® Core™ i7-11850H @ 2.50GHz x 16, 32GB Memory). In read experiments on the same dataset the speedup is ~ x 2.4, and that does not account that HDF5 cannot cache automatically portion of the data, as RocksDB can. See also [ssg2/Notebooks/HDF5_vs_RocksDB.ipynb](ssg2/Notebooks/HDF5_vs_RocksDB.ipynb) for the benchmarks, and to run your own tests. 
