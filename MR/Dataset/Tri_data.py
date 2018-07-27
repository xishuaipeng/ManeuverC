from Dataset.Base_data import Base_data as Base_data  
import os
import scipy.io as sio
import csv
import numpy as np
import h5py
import hdf5storage
class Tri_data (Base_data): 

    def __init__( self, data_dir, session_id_list,  quiet = False):
        self.sid_list = session_id_list
        self.data_dir = data_dir
        self.quiet = quiet
        self.event_list = []
        self.event_list_shuff = []
        self.n_event = 0
        self.loop_evnt = 0
        self.loop_seq = 0
    
    def print(self, message):
        if not self.quiet:
            print(message)

    def encoder(self):
        for sid in self.sid_list:
            evt_path = os.path.join(self.data_dir,sid,'Pydata') 
            evt_list = os.listdir(evt_path)
            evt_list = list(map(lambda x, y=evt_path: os.path.join(y,x), evt_list) ) 
            evt_list = list(filter(lambda x:  os.path.isdir(x),evt_list))
            self.print(evt_list)
            self.event_list .extend(evt_list)     
            self.print('Find Total %d events in dir %s'%(len(self.event_list), evt_path ))
            self.print(evt_list)
        self.n_event = len(self.event_list)


    def decoder(self,data):
        data_field = ['frame', 'signal','path','label','session']
        frame_list = np.squeeze(data['frame'].astype(np.int)).tolist()
        frame_path = ''.join(data['path'][0][0][0])
        #frame_path = self.data_dir
        front_img_list = list(map(lambda x, y = frame_path: os.path.join(y,'%d_f.jpg'%x) , frame_list))
        driver_img_list = list(map(lambda x, y = frame_path: os.path.join(y,'%d_d.jpg'%x) , frame_list))
        signal = np.squeeze(data['signal']) 
        label = np.squeeze(data['label']['sequenceLabel']).tolist()
        data = {'fimg':front_img_list,'dimg':driver_img_list,'signal': signal, 'label':label}
        return data



    def next(self, reset=False, shuffle = False):
        # if reset:
        #     self.loop_evnt = 0
        #     self.loop_seq = 0
        #     self.event_list_shuff = self.event_list
        if self.event_list_shuff == []:
            self.event_list_shuff =  self.event_list
            if shuffle:
                np.random.shuffle( self.event_list_shuff)
            self.loop_evt = 0
            self.loop_seq = 1
        assert(len(self.event_list_shuff)>0)
        event_dir  = self.event_list_shuff[self.loop_evt]
        seq_mat = os.path.join(event_dir, '%d.mat'%self.loop_seq)
        if os.path.isfile(seq_mat):
            self.print(seq_mat)
            data = hdf5storage.loadmat(seq_mat)['seq_struct']
            self.loop_seq = self.loop_seq + 1
            return self.decoder(data), False
        else:
            self.loop_evt = (self.loop_evt +1) #% len( self.event_list_shuff)+1
            self.loop_seq = 1
            if self.loop_evt >= self.n_event:
                self.event_list_shuff = []
                return None, True
            return self.next(reset=reset, shuffle = shuffle)
    


        










