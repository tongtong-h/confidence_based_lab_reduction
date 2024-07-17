import torch
from torch import nn
import torch.nn.functional as F
import math


# +
class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=1500):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        
    def forward(self, t):
        return self.lin(self.emb(t))


class GRU_Attention(torch.nn.Module):
    # Feed-forward attention mechanism: https://arxiv.org/abs/1512.08756
    def __init__(self, 
                 input_dim,
                 hidden_dim):
        super(GRU_Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.embedding_layer = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.W_hidden = nn.Linear(hidden_dim, 1)

    def embedding(self, x_t):
        x_out, hidden_out = self.embedding_layer(x_t)
    
        return x_out, hidden_out

    def activation(self, h_t):
        h_out = self.W_hidden(h_t)
        
        return h_out

    def attention(self, e_t):
        softmax = torch.nn.Softmax(dim=1)
        alphas = softmax(e_t)
        
        return alphas

    def context(self, alpha_t, x_t):
        return torch.bmm(alpha_t.transpose(1,2), x_t)


    def forward(self, x):
        time_len = x.shape[1]
        c_out, alpha_out = [], []
        
        for t in range(time_len):
            x_e, hidden_e = self.embedding(x[:,:t+1])
            x_a = self.activation(x_e)
#             x_a = self.activation(hidden_e)
            alpha = self.attention(x_a)
            x_c = self.context(alpha, x_e)
            
            c_out.append(x_c)
            alpha_out.append(alpha)
        
        c_out = torch.cat(c_out, 1)
        alpha_out = torch.cat(alpha_out, 1)
        
        return c_out, alpha_out


# https://arxiv.org/pdf/1605.05101.pdf
class LSTM_Attention_model(nn.Module):
    def __init__(self, 
                 param_args,
                 lstm_test_dim=64,
                 static_dim=6,
                 is_bidirectional=False,
                 time_dim=10,
                 pooling='last'):
        super(LSTM_Attention_model, self).__init__()
        self.num_panel = param_args['test_num'] # 12
        self.len_time = param_args['len_time'] # 30
        self.lstm_test_dim = lstm_test_dim
        self.pooling = pooling
        self.is_bidirectional = is_bidirectional
        
        self.rte = RelTemporalEncoding(n_hid=time_dim)
        
        self.shared_layer = nn.GRU(self.num_panel*2+static_dim+time_dim,self.lstm_test_dim)
        self.task_layer1 = GRU_Attention(1+self.lstm_test_dim+static_dim+self.num_panel+time_dim,
                                   self.lstm_test_dim)
        self.task_layer2 = GRU_Attention(self.num_panel+self.lstm_test_dim+static_dim+self.num_panel+time_dim,
                                   self.lstm_test_dim)
        
    
    def forward(self,input_batch, hgb_delta, test_time_delta, static_feature, nnan_mask): #B L 12       
        test_time_delta = self.rte(test_time_delta.long().squeeze(2))
        
        shared_lstm_out,_ = self.shared_layer(torch.cat([input_batch,nnan_mask,
                                                         static_feature,
                                                         test_time_delta],2))

        # stable
        task_out1, attention1 = self.task_layer1(torch.cat([shared_lstm_out,nnan_mask,
                                                            hgb_delta,
                                                            static_feature,
                                                            test_time_delta],2))
        # normal, select
        task_out2, attention2 = self.task_layer2(torch.cat([shared_lstm_out, nnan_mask,
                                                            hgb_delta,
                                                            input_batch,
                                                            static_feature,
                                                            test_time_delta],2))
        

        return task_out1, task_out2