# -*- coding: utf-8 -*-
# +
import torch
from torch import nn
import torch.nn.functional as F

from network import LSTM_Attention_model


# -

class TS_Feature_Extractor(torch.nn.Module):
    def __init__(self, 
                 param_args,
                 embedding_size):
        super(TS_Feature_Extractor, self).__init__()
        
        self.param_args = param_args
        self.len_time = param_args['len_time'] - 1
        self.device = param_args['device']
        self.num_feature = param_args['test_num']
        self.num_lab_feature = param_args['lab_test_num']
        self.person_input_dim = param_args['person_feature_dim']
        self.person_output_dim = 6
        self.embedding_size = embedding_size
        
        self.input_size = self.num_feature*2 + self.person_output_dim + 1
        
        self.person_linear = torch.nn.Linear(self.person_input_dim, self.person_output_dim)
            
        self.encoder_ts = LSTM_Attention_model(self.param_args, lstm_test_dim=self.embedding_size)


    def forward(self, 
                input_batch, 
                hgb_delta,
                test_time_delta, 
                person_info, 
                nnan_mask):
        
        time_len = input_batch.shape[1]
        person_info = person_info.float()
        person_feature = self.person_linear(person_info)
        person_feature = person_feature.repeat(time_len,1,1).transpose(0,1)
        static_feature = person_feature
        
        input_batch_clone = input_batch.clone()
        if self.training:
            size_o = input_batch[:,:,:self.num_lab_feature].size()
            corrupt = torch.rand(size_o) + 0.95
            corrupt = corrupt.to(self.param_args['device'])
            corrupt[corrupt>=1] = 1
            corrupt[corrupt<1] = 0
            self.corrupt = corrupt
            input_batch_clone[:,:,:self.num_lab_feature] = corrupt * input_batch_clone[:,:,:self.num_lab_feature]
        
        
        ts_out = self.encoder_ts(input_batch,
                                 hgb_delta,
                                 test_time_delta, 
                                 person_feature, 
                                 nnan_mask)
        
        return ts_out


class Selective_Forecaster(torch.nn.Module):
    def __init__(self, 
                 param_args
                ):
        super(Selective_Forecaster, self).__init__()
        
        self.param_args = param_args
        self.embedding_size = 128
        self.pred_hidden_dim = 64
        self.num_out = 1
        
        self.ts_feature_extractor = TS_Feature_Extractor(param_args, self.embedding_size)
            

        self.aux_value_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pred_hidden_dim, self.num_out),
            torch.nn.ReLU()
        )
        

        self.select_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pred_hidden_dim, self.num_out),
            torch.nn.Sigmoid()
        )
        
        self.normal_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pred_hidden_dim, self.num_out),
        )
    
        self.stable_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.pred_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.pred_hidden_dim, self.num_out),
        )


    def forward(self, input_batch, hgb_delta,
                test_time_delta,
                person_info, nnan_mask):

        ts_out1, ts_out2 = self.ts_feature_extractor(input_batch, 
                                                     hgb_delta,
                                                     test_time_delta,
                                                     person_info, nnan_mask)
        
        stable_out = self.stable_out(ts_out1)
        aux_out = self.aux_value_out(ts_out2)
        select_out = self.select_out(ts_out2)
        normal_out = self.normal_out(ts_out2)

        return aux_out, normal_out, stable_out, select_out

