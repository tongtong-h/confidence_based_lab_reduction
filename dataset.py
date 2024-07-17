import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class LabDataset(Dataset):    
    def __init__(self, lab_test_data, non_nan_mask, person_feature, PARAMS):
        self.icustay_id = lab_test_data.index.unique().tolist()
        self.test_data_all = torch.FloatTensor(lab_test_data[lab_columns+vital_columns].values.reshape(-1,PARAMS['len_time'], PARAMS['measure_num']))
        self.hgb_delta = torch.FloatTensor(lab_test_data[['Hgb_delta']].values.reshape(-1,PARAMS['len_time'],1))
        self.non_nan_mask = torch.FloatTensor(non_nan_mask[lab_columns+vital_columns].values.reshape(-1,PARAMS['len_time'],PARAMS['measure_num']))
        self.normal_mask = torch.FloatTensor(lab_test_data[['Hgb_normal_mask']].values.reshape(-1,PARAMS['len_time'],1)
        self.stable_mask = torch.FloatTensor(lab_test_data[['Hgb_stable_mask']].values.reshape(-1,PARAMS['len_time'],1)
        self.delta_time = torch.FloatTensor(lab_test_data['time_delta'].values.reshape(-1,PARAMS['len_time'],1))
        
        self.person_feature = torch.FloatTensor(person_feature.values.reshape(-1,len(demo_columns)))     

        self.normal_low = torch.FloatTensor(lab_test_data[['Hgb_normal_low']].values.reshape(-1,PARAMS['len_time'],1))
        self.normal_high = torch.FloatTensor(lab_test_data[['Hgb_normal_high']].values.reshape(-1,PARAMS['len_time'],1))

        # Normalization 
        scaler = StandardScaler()
        lab_test_features = lab_test_data[lab_columns+vital_columns].values
        lab_test_features[lab_test_features==0] = np.nan
        scaler.fit(lab_test_features)
        lab_test_features = scaler.transform(lab_test_features)
        lab_test_features = np.nan_to_num(lab_test_features)
        self.scaler = scaler
        self.test_data_scale = torch.FloatTensor(lab_test_features.reshape(-1,PARAMS['len_time'], 
                                                                           PARAMS['measure_num']))
        
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)                         
    
    def __len__(self):
        return len(self.icustay_id)

    def __getitem__(self, idx):
        icustay_id = self.icustay_id[idx]
        test_data_all = self.test_data_all[idx]
        test_data_scale = self.test_data_scale[idx]
        hgb_delta = self.hgb_delta[idx]
        non_nan_mask = self.non_nan_mask[idx]
        normal_mask = self.normal_mask[idx]
        stable_mask = self.stable_mask[idx]
        delta_time = self.delta_time[idx]
        person_feature = self.person_feature[idx]
        normal_low = self.normal_low[idx]
        normal_high = self.normal_high[idx]
        
        sample = {'icustay_id': icustay_id, 
                  'test_data_all': test_data_all,
                  'test_data_scale': test_data_scale,
                  'hgb_delta': hgb_delta,
                  'non_nan_mask': non_nan_mask,
                  'normal_mask': normal_mask,
                  'stable_mask': stable_mask,
                  'delta_time': delta_time,
                  'person_feature': person_feature,
                  'normal_low': normal_low,
                  'normal_high': normal_high
                 }
        
        return sample
