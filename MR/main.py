from  Dataset.Tri_data import Tri_data
data_list = ['ID001_T002']# 'ID001_T001',
data_dir = 'D:\Develop\Project\TRI_Manuver\Out'
import hdf5storage
import collections as cl
import numpy as np

import scipy.io as sio


# x = hdf5storage.loadmat('D:\Develop\Project\TRI_Manuver\Out\ID001_T001\pmat.mat')
# h=5
data = Tri_data(data_dir , data_list, quiet=False)
data.encoder()
for i in range(50):
    slice_data = data.next(shuffle=True)
    print(slice_data['img'])