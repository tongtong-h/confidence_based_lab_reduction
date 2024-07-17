# +
from utils import read_config_dict

cfg = read_config_dict()

# +
import os
import random
import pickle

import numpy as np
import pandas as pd
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from dataset import LabDataset
from trainer import Trainer
# -

if cfg['device'] == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
    torch.cuda.set_device(0)

# +
seed = cfg['seed']
torch.manual_seed(seed)

if device.type == 'cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# +
lab_columns = sorted(['BUN', 'Calcium',
                      'Chloride', 'Creatinine', 'HCO3', 'Hgb', 'Magnesium',
                      'Phosphate', 'Platelet', 'Potassium', 'Sodium', 'WBC'])

vital_columns = sorted(['diasbp', 'heartrate', 'meanbp', 'resprate','spo2', 'sysbp'])

demo_columns = sorted(['gender_code','race_code','admission_age'])


PARAMS = {
    # Fixed hyparameters
    'len_time': cfg['len_time'],
    'lab_test_num': len(lab_columns),
    'measure_num': len(lab_columns)+len(vital_columns),
    'demo_num': len(demo_columns),
    'hgb_index': [lab_columns.index('Hgb')],
    'device': torch.device(cfg['device']),
    # Controlled hyperparameters
    'kfold': cfg['kfold'],
    'debug': cfg['debug'],
    'epochs': cfg['epochs'],
    'lr': cfg['lr'],
    'batch_size': cfg['batch_size'],
    'coverage': cfg['coverage'],
    'input_dir': cfg['input_dir'],
    'model_name': cfg['model_name'],
    'model_path': cfg['output_dir']+cfg['model_name']+'_c'+str(cfg['coverage'])+'_kf'+str(cfg['kfold'])+'.md',
}


# +
train_dir = PARAMS['input_dir']

lab_test = pd.read_csv(train_dir + 'lab_test_data.csv')
non_nan_mask = pd.read_csv(train_dir + 'non_nan_mask.csv')
# -

if PARAMS['debug']:
    small_samples = lab_test['icustay_id'].unique()[:150]
    lab_test = lab_test[lab_test['icustay_id'].isin(small_samples)]
    non_nan_mask = non_nan_mask[non_nan_mask['icustay_id'].isin(small_samples)]


# +
correspond_id_all = np.array(lab_test['icustay_id'].unique())
np.random.shuffle(correspond_id_all)

kf = KFold(n_splits=5)
for i, (train_index, test_index) in enumerate(kf.split(correspond_id_all)):
    if i == PARAMS['kfold']:
        valid_length = len(train_index)//8
        train_length = len(train_index) - valid_length
        
        train_icuid_list = correspond_id_all[train_index[:train_length]]
        valid_icuid_list = correspond_id_all[train_index[train_length:train_length+valid_length]]
        test_icuid_list = correspond_id_all[test_index]
        
        print('Data split:', train_length, valid_length, len(test_index))

train_lab_test = lab_test[lab_test['icustay_id'].isin(train_icuid_list)]
train_non_nan_mask = non_nan_mask[non_nan_mask['icustay_id'].isin(train_icuid_list)]

valid_lab_test = lab_test[lab_test['icustay_id'].isin(valid_icuid_list)]
valid_non_nan_mask = non_nan_mask[non_nan_mask['icustay_id'].isin(valid_icuid_list)]

test_lab_test = lab_test[lab_test['icustay_id'].isin(test_icuid_list)]
test_non_nan_mask = non_nan_mask[non_nan_mask['icustay_id'].isin(test_icuid_list)]

# +
train_person_f = train_lab_test[['icustay_id']+demo_columns].groupby('icustay_id').first()
valid_person_f = valid_lab_test[['icustay_id']+demo_columns].groupby('icustay_id').first()
test_person_f = test_lab_test[['icustay_id']+demo_columns].groupby('icustay_id').first()

train_lab_test.set_index('icustay_id', inplace=True)
train_non_nan_mask.set_index('icustay_id', inplace=True)
valid_lab_test.set_index('icustay_id', inplace=True)
valid_non_nan_mask.set_index('icustay_id', inplace=True)
test_lab_test.set_index('icustay_id', inplace=True)
test_non_nan_mask.set_index('icustay_id', inplace=True)
# -

if PARAMS['split_valid_data'] is False:
    train_lab_test = pd.concat([train_lab_test, valid_lab_test])
    train_non_nan_mask = pd.concat([train_non_nan_mask, valid_non_nan_mask])
    train_person_f = pd.concat([train_person_f, valid_person_f])

# +
train_dataset = LabDataset(train_lab_test, train_not_nan_mask, 
                           train_person_f, PARAMS)
valid_dataset = LabDataset(valid_lab_test, valid_not_nan_mask, 
                           valid_person_f, PARAMS)
test_dataset = LabDataset(test_lab_test, test_not_nan_mask, 
                          test_person_f, PARAMS)

train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=PARAMS['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=PARAMS['batch_size'], shuffle=False, pin_memory=True)


# +
def run_experiments(train_loader, valid_loader, test_loader,
                    save_results=True):
    model = Trainer(
        param_args=PARAMS
    )
    

    model.fit_model(train_loader, valid_loader, epochs=PARAMS['epochs'], lr=PARAMS['lr'])
    
    mode_list = ['reduction_all','no_reduction']
    
    for reduction_mode in mode_list:
        print(f'reduction_mode = {reduction_mode}')
        test_pred_results = model.predict(test_loader, 
                                          reduction_mode=reduction_mode)
        
        if save_results:
            model_name = PARAMS['model_name']
            coverage_rate = PARAMS['coverage']
            kfold = PARAMS['kfold']
            
            with open(f'output_ml/{model_name}_test_pred_c{coverage_rate}_kf{kfold}_{reduction_mode}.pkl', 'wb') as f:
                pickle.dump(test_pred_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_test_ground_truth(test_loader, name='test'):
    ids = []
    targets, hgb_nnan_mask = [], []
    normal_mask, stable_mask = [], []

    for sample in test_loader:
        ids.append(sample['icustay_id'])
        targets.append(sample['test_data_all'][:,:,PARAMS['hgb_index']])
        hgb_nnan_mask.append(sample['not_nan_mask'][:,past_split:,PARAMS['hgb_index']])  
        normal_mask.append(sample['normal_mask'])
        stable_mask.append(sample['stable_mask'])

        
    ids = torch.cat(ids)
    targets = torch.cat(targets)
    hgb_nnan_mask = torch.cat(hgb_nnan_mask)
    normal_mask = torch.cat(normal_mask)
    stable_mask = torch.cat(stable_mask)
    
    ground_truth = {
        'IDs': ids.cpu(),
        'Targets': targets.cpu(),
        'Not nan mask': hgb_nnan_mask.cpu(),
        'Normal mask': normal_mask.cpu(),
        'Stable mask': stable_mask.cpu()
    }
    
    kfold = PARAMS['kfold']
    with open(f'output_ml/{name}_ground_truth_kf{kfold}.pkl', 'wb') as f:
        pickle.dump(ground_truth, f, protocol=pickle.HIGHEST_PROTOCOL)

# +
save_test_ground_truth(test_loader, 'test')

test_pred_results = run_experiments(train_loader,
                                    valid_loader,
                                    test_loader,
                                    save_results=True)
