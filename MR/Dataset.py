
import os
import scipy.io as sio
import csv
import numpy as np
import re
import h5py
import hdf5storage

class Base_data:
    def __init__(self):
        print('NO IMPLEMENT')
    
    def _encode(self,**kwargs):
        print('NO IMPLEMENT')

    def _decode(self, **kwargs):
        print('NO IMPLEMENT')
    
    def print(self, message):
        if not self.quiet:
            print(message)
    def _hashfun(self, index):
        assert index < self.n_seq
        (eindex, seq_top) = list(filter(lambda x, y = index: x[1] > y,enumerate(self.hash_list)))[0]
        rev_seq_index = seq_top - index
        return eindex, rev_seq_index


class TRI (Base_data): 

    def __init__( self, **kwargs):
        data_dir = ''
        session_id_list=[]
        quiet = False
        for key, value in kwargs.items():
            if key == 'data_dir':
                data_dir = value
            if key == 'data_list':
                session_id_list = value
            if key == 'quiet':
                quiet = value
        assert(not data_dir == '') &(not len(session_id_list)==0)
        self.sid_list = session_id_list
        self.data_dir = data_dir
        self.quiet = quiet
        self.loop_list = []
        self.evt_seq = []
        self.n_evt = 0
        self.n_seq = 0
        self.hash_list = []
        self.ncase = 0

    def _encode(self, **kwargs):
        unit = 'seq'
        for key, value in kwargs.items():
            if key == 'unit':
                unit = value
        for sid in self.sid_list:
            event_list = []
            evt_path = os.path.join(self.data_dir,sid,'Pydata') 
            evt_list = os.listdir(evt_path)
            evt_list = list(map(lambda x, y=evt_path: os.path.join(y,x), evt_list) ) 
            evt_list = list(filter(lambda x:  os.path.isdir(x),evt_list))
            evt_list = sorted(evt_list, key = lambda x: int(re.findall('\d+',x)[-1]) )
            # event_list.extend(evt_list)     
            self.print('Find total %d events in session-%s'%(len(evt_list), sid ))
            self.n_evt = self.n_evt + len(evt_list)
            for event_dir in evt_list:
                seq_list = os.listdir(event_dir)
                seq_list = list(map(lambda x, y=event_dir: os.path.join(y,x), seq_list) ) 
                seq_list = list(filter(lambda x:  x.endswith('.mat'),seq_list))
                seq_list = sorted(seq_list, key = lambda x: int(re.findall('\d+',x)[-1]))
                self.evt_seq.append(seq_list)
                self.print('Find total %d sequence in event-%s'%(len(seq_list), event_dir ))
                self.n_seq = self.n_seq + len(seq_list)
                self.hash_list.extend([self.n_seq])
        self.print('Total event: %d, Total sequence:%d'%(self.n_evt, self.n_seq))


    def _decode(self,**kwargs):
        mat_path = ''
        for key, value in kwargs.items():
            if key == 'mat_path':
                mat_path = value
        assert (not mat_path=='')
        assert (os.path.isfile(mat_path) & mat_path.endswith('mat'))
        data = hdf5storage.loadmat(mat_path)['seq_struct']
        data_field = ['frame', 'signal','path','label','session']
        frame_list = np.squeeze(data['frame'].astype(np.int)).tolist()
        frame_path = ''.join(data['path'][0][0][0])
        #frame_path = self.data_dir
        front_img_list = list(map(lambda x, y = frame_path: os.path.join(y,'%d_f.jpg'%x) , frame_list))
        driver_img_list = list(map(lambda x, y = frame_path: os.path.join(y,'%d_d.jpg'%x) , frame_list))
        signal = np.squeeze(data['signal']) 
        label = np.squeeze(data['label']['sequenceLabel']).tolist()
        dic = {'fimg':front_img_list,'dimg':driver_img_list,'signal': signal, 'label':label}
        return dic



    def next(self, reset=False, shuffle = False, unit = 'seq'):
        if self.n_evt == 0:
            self._encode(unit=unit)
        assert (self.n_evt > 0) & (self.n_seq > 0)
        if len(self.loop_list)==0 | reset:
            if unit =='seq':
                self.loop_list = np.arange(self.n_seq)
            else:
                self.loop_list = np.arange(self.n_evt)
            if shuffle:
                np.random.shuffle(self.loop_list)
            self.loop_list  = list(self.loop_list )

        cur_index = self.loop_list.pop()
        if unit =='seq':
            evt_index, re_seq_index = self._hashfun(cur_index)
            seq_list = self.evt_seq[evt_index]
            cur_path = [seq_list[-re_seq_index]]
        else:
            cur_path = self.evt_seq[cur_index]
        
        loop_data = []
        for mat_path in cur_path:
            loop_data.append(self._decode(mat_path = mat_path))
        return  loop_data, len(self.loop_list)==0 

def data_factory(name, **kwargs):
    if name == 'TRI':
        return TRI(**kwargs)

    


        










