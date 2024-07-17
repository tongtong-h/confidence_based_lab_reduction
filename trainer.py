# -*- coding: utf-8 -*-
# +
import numpy as np

import torch
import torch.nn.functional as F

from network import Selective_Forecaster
from utils import get_metrics


# -

class Trainer():
    def __init__(self, param_args):
        super().__init__()
        
        self.param_args = param_args
        self.coverage_rate = param_args['coverage']
        self.device = param_args['device']
        self.path = param_args['model_path']
        
        self.model = Selective_Forecaster(param_args)
        

    def fit_model(self, train_loader, valid_loader, epochs, lr, patience=10):
        if os.path.isfile(self.path):
            self.model = torch.load(self.path, map_location=self.device)
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = self.model.to(self.device)
            self.train(train_loader, valid_loader, epochs, lr, patience)
            # Load best checkpoint
            self.model = torch.load(self.path, map_location=self.device)
            for param in self.model.parameters():
                param.requires_grad = False
    
       

    def loss_func(
        self,
        select_out, normal_out, stable_out, aux_out,
        batch_nnan_mask,
        batch_normal_mask, batch_stable_mask, 
        targets, 
        batch_normal_range
    ): 
        
        bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        aux_loss, select_loss = 0.0, 0.0, 0.0
        normal_loss, stable_loss = 0.0, 0.0
        
        sample_loss = 0.0
        sample_select_loss = 0.0
        sample_aux_loss = 0.0, 0.0
        sample_normal_loss, sample_stable_loss = 0.0, 0.0
        
        epsilon = 1e-10

        batch_obs_mask = batch_stable_mask
            
        lab_select_mean = torch.mean(select_out[batch_obs_mask>=0])

        lab_normal_loss = bce_criterion(normal_out,batch_normal_mask)[batch_obs_mask>=0]
        normal_loss = (lab_normal_loss *
                       select_out[batch_obs_mask>=0]).mean() / (lab_select_mean+epsilon)
        sample_normal_loss = lab_normal_loss.mean().detach().item()

        lab_stable_loss = bce_criterion(stable_out,batch_stable_mask)[batch_obs_mask>=0]
        stable_loss = (lab_stable_loss *
                      select_out[batch_obs_mask>=0]).mean() / (lab_select_mean+epsilon)
        sample_stable_loss = lab_stable_loss.mean().detach().item()
        
        lab_aux_loss = (aux_out - targets[:,1:])
        lab_aux_loss = ((lab_aux_loss / batch_normal_range[:,1:])**2)[batch_obs_mask>=0]
        sample_aux_loss = lab_aux_loss.detach().mean().item()
        aux_loss = lab_aux_loss.mean()
            

        penalty_lambda = 32
        penalty = penalty_lambda*torch.max(torch.tensor([0.0]).to(self.device), 
                                           self.coverage_rate-lab_select_mean)**2
        select_loss = penalty
        sample_select_loss = penalty.detach().item()

        sample_select_mean = lab_select_mean.detach().cpu().item()

        self.alpha = 0.8
        
        loss = self.alpha * (normal_loss+stable_loss+
                             select_loss) + (1-self.alpha) * aux_loss
        
        return (
            loss, normal_loss, stable_loss, select_loss,
            aux_loss, 
            sample_normal_loss, sample_stable_loss, 
            sample_select_loss, sample_aux_loss, 
            sample_select_mean
        )
            
      
    def train(self, train_loader, valid_loader, epochs, lr, patience=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=10e-7)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
        
        best_loss = np.inf

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_select_loss = 0.0
            train_aux_loss = 0.0
            train_normal_loss, train_stable_loss = 0.0, 0.0
            train_select_mean = 0.0

            for idx, data in enumerate(train_loader):                    
                optimizer.zero_grad()
                batch_input_data = data['test_data_all'].to(self.device)
                batch_input_data_scale = data['test_data_scale'].to(self.device)
                batch_hgb_delta = data['hgb_delta'].to(self.device)
                batch_pfeature = data['person_feature'].to(self.device)
                batch_nnan_mask = data['non_nan_mask'].to(self.device)
                batch_normal_mask = data['normal_mask'].to(self.device)
                batch_stable_mask = data['stable_mask'].to(self.device)
                batch_delta_time = data['delta_time'].to(self.device)
                batch_normal_high = data['normal_high'].to(self.device)
                batch_normal_low = data['normal_low'].to(self.device)


                (aux_out, 
                 normal_out, stable_out, 
                 select_out) = self.model(
                    batch_input_data_scale[:,:-1],
                    batch_hgb_delta[:,:-1],
                    batch_delta_time[:,:-1],
                    batch_pfeature,
                    batch_nnan_mask[:,:-1],
                )
                
                targets = batch_input_data[:,:,self.param_args['hgb_index']]
                batch_nnan_mask = batch_nnan_mask[:,:,self.param_args['hgb_index']]
                batch_normal_mask = batch_normal_mask[:,1:]
                batch_stable_mask = batch_stable_mask[:,1:]
                
                batch_normal_range = batch_normal_high - batch_normal_low
                
                
                (
                    loss, normal_loss, stable_loss, 
                    select_loss, aux_loss, 
                    sample_normal_loss, sample_stable_loss, 
                    sample_select_loss, 
                    sample_aux_loss, 
                    sample_select_mean
                ) = self.loss_func(
                        select_out, normal_out, stable_out, aux_out,
                        batch_nnan_mask,
                        batch_normal_mask, batch_stable_mask, 
                        targets, 
                        batch_normal_range
                )
                

                loss.backward()
                optimizer.step()
                
                if idx % 20 == 0:
                    print(f'Iter: {idx}\tTrain loss: {loss.item()}' +
                          f' Select loss: {sample_select_loss}'+
                          f' Aux loss: {sample_aux_loss}'+
                          f' Normal loss: {sample_normal_loss}' +
                          f' Stable loss: {sample_stable_loss}' +
                          f' Select mean: {sample_select_mean}'
                         )
                
                train_loss += loss.item()
                train_select_loss += sample_select_loss
                train_normal_loss += sample_normal_loss
                train_stable_loss += sample_stable_loss
                train_aux_loss += sample_aux_loss
                train_select_mean += sample_select_mean
            
            mean_train_loss = train_loss / len(train_loader)
            print(f'Epoch: {epoch}\tTrain loss: {mean_train_loss}' +
                  f' Select loss: {train_select_loss / len(train_loader)}'+
                  f' Aux loss: {train_aux_loss / len(train_loader)}'+
                  f' Normal loss: {train_normal_loss / len(train_loader)}' +
                  f' Stable loss: {train_stable_loss / len(train_loader)}' +
                  f' Select mean: {train_select_mean / len(train_loader)}'
                 )

            
            mean_valid_loss = self.validate(valid_loader)
            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                if self.param_args['debug'] == False:
                    torch.save(self.model, self.path)
            print(f'Best loss: {best_loss}')
    
    
    @torch.no_grad()
    def validate(self, valid_loader):
        best_loss = np.inf

        self.model.eval()
        valid_loss = 0.0
        valid_select_loss = 0.0
        valid_aux_loss = 0.0
        valid_normal_loss, valid_stable_loss = 0.0, 0.0
        valid_select_mean = 0.0
        
        normal_pred_all, normal_mask_all = [], []
        stable_pred_all, stable_mask_all = [], []
        select_pred_all = []
        
        for idx, data in enumerate(valid_loader):
            batch_input_data = data['test_data_all'].to(self.device)
            batch_input_data_scale = data['test_data_scale'].to(self.device)
            batch_hgb_delta = data['hgb_delta'].to(self.device)
            batch_pfeature = data['person_feature'].to(self.device)
            batch_nnan_mask = data['non_nan_mask'].to(self.device)
            batch_normal_mask = data['normal_mask'].to(self.device)
            batch_stable_mask = data['stable_mask'].to(self.device)
            batch_delta_time = data['delta_time'].to(self.device)
            batch_normal_low = data['normal_low'].to(self.device)
            batch_normal_high = data['normal_high'].to(self.device)

            (aux_out, 
             normal_out, stable_out, 
             select_out) = self.model(
                batch_input_data_scale[:,:-1],
                batch_hgb_delta[:,:-1],
                batch_delta_time[:,:-1],
                batch_pfeature, 
                batch_nnan_mask[:,:-1]
            )
                
            targets = batch_input_data[:,:,self.param_args['hgb_index']]
            batch_nnan_mask = batch_nnan_mask[:,:,self.param_args['hgb_index']]
            batch_normal_mask = batch_normal_mask[:,1:]
            batch_stable_mask = batch_stable_mask[:,1:]

            batch_normal_range = batch_normal_high - batch_normal_low
            
            normal_pred_all.append(normal_out)
            stable_pred_all.append(stable_out)
            normal_mask_all.append(batch_normal_mask)
            stable_mask_all.append(batch_stable_mask)
            select_pred_all.append(select_out)
            
            (
                _, normal_loss, stable_loss, 
                select_loss, aux_loss, 
                sample_normal_loss, sample_stable_loss,
                sample_select_loss, 
                sample_aux_loss, 
                sample_select_mean
            ) = self.loss_func(
                    select_out, normal_out, stable_out, aux_out,
                    batch_nnan_mask,
                    batch_normal_mask, batch_stable_mask, 
                    targets, 
                    batch_normal_range
            )
            

            valid_loss += (normal_loss+stable_loss).item()
            valid_select_loss += sample_select_loss
            valid_normal_loss += sample_normal_loss
            valid_stable_loss += sample_stable_loss
            valid_aux_loss += sample_aux_loss
            valid_select_mean += sample_select_mean
        
        mean_valid_loss = valid_loss / len(valid_loader)
        print(f'Validation loss: {mean_valid_loss}' +
              f' Select loss: {valid_select_loss / len(valid_loader)}'+
              f' Aux loss: {valid_aux_loss / len(valid_loader)}'+ 
              f' Normal loss: {valid_normal_loss / len(valid_loader)}' +
              f' Stable loss: {valid_stable_loss / len(valid_loader)}' +
              f' Select mean: {valid_select_mean / len(valid_loader)}'
             )
        if valid_select_mean / len(valid_loader) < 0.05:
            mean_valid_loss = 10000
        
        normal_pred_all = torch.cat(normal_pred_all)
        normal_mask_all = torch.cat(normal_mask_all)
        stable_pred_all = torch.cat(stable_pred_all)
        stable_mask_all = torch.cat(stable_mask_all)
        select_pred_all = torch.cat(select_pred_all)
        

        select_mask = (select_pred_all>=0.5)&(stable_mask_all>=0)
        
        if select_mask.sum() > 0:
            print('Normal', get_metrics(torch.sigmoid(normal_pred_all)[select_mask],
                                        normal_mask_all[select_mask]))
            print('Stable', get_metrics(torch.sigmoid(stable_pred_all)[select_mask], 
                                        stable_mask_all[select_mask]))
        
        return mean_valid_loss
    
    
    @torch.no_grad()
    def predict_batch(self, data, reduction_mode='all', score_thres=0.5, past_split=0):
        batch_input_data = data['test_data_all'].to(self.device)
        batch_input_data_scale = data['test_data_scale'].to(self.device)
        batch_hgb_delta = data['hgb_delta'].to(self.device)
        batch_pfeature = data['person_feature'].to(self.device)
        batch_nnan_mask = data['non_nan_mask'].to(self.device)
        batch_normal_mask = data['normal_mask'].to(self.device)
        batch_stable_mask = data['stable_mask'].to(self.device)
        batch_delta_time = data['delta_time'].to(self.device)
        
        new_input_data_scale, new_nnan_mask, new_mfeature = None, None, None

        select_out = []
        normal_out, stable_out = [], []
        reduce_mask, auto_mask = [], []
        
        if reduction_mode == 'no_reduction':
            (_, 
             normal_out, stable_out, 
             select_out) = self.model(
                batch_input_data_scale[:,:-1], 
                batch_hgb_delta[:,:-1],
                batch_delta_time[:,:-1],
                batch_pfeature,
                batch_nnan_mask[:,:-1]
            )
            
            normal_out = torch.sigmoid(normal_out)
            stable_out = torch.sigmoid(stable_out)
            
            select_group = torch.where(select_out>=score_thres, 1.0, 0.0)
            select_group = select_group.squeeze(-1)
            select_group[batch_stable_mask[:,1:,0]==-1] = np.nan
            
            reduce_mask = (select_group>0)
            auto_mask = (select_group>0)
        
        else:
            new_input_data_scale = batch_input_data_scale[:,:past_split+1]
            new_hgb_delta = batch_hgb_delta[:,:past_split+1],
            new_nnan_mask = batch_nnan_mask[:,:past_split+1]

            # next lab is at t=past_split+1
            (_, 
             t_normal_out, t_stable_out, 
             t_select_out) = self.model(
                new_input_data_scale, 
                new_hgb_delta,
                batch_delta_time[:,:past_split+1],
                batch_pfeature,
                new_nnan_mask
            )
    
            normal_out.append(torch.sigmoid(t_normal_out))
            stable_out.append(torch.sigmoid(t_stable_out))
            

            for t in range(past_split+1, self.param_args['len_time']):
                t_input_data_scale = batch_input_data_scale[:,t:t+1].clone()
                t_hgb_delta = batch_hgb_delta[:,t:t+1].clone()
                t_nnan_mask = batch_nnan_mask[:,t:t+1].clone()

                t_normal_mask = batch_normal_mask[:,t:t+1].clone()
                t_stable_mask = batch_stable_mask[:,t:t+1].clone()

                # reduce Hgb if predictable (select_prob>=0.5)
                tt_select_out = select_out[-1].clone()
                t_select_group = torch.where(tt_select_out>=score_thres, 1.0, 0.0)

                tt_normal_out = normal_out[-1].clone()
                tt_stable_out = stable_out[-1].clone()

                t_auto_mask = (t_select_group>0)
        
                if reduction_mode == 'reduction_all':
                    t_reduce_mask = (t_auto_mask>0)&(tt_normal_out>=0.5)&(tt_stable_out>=0.5)
#                     t_reduce_mask = t_reduce_mask.squeeze(-1)
                elif reduction_mode == 'reduction_select':
                    t_reduce_mask = (t_auto_mask>0)
#                     t_reduce_mask = t_reduce_mask.squeeze(-1)

                reduce_mask.append(t_reduce_mask)
                auto_mask.append(t_auto_mask)

                # No need to predict timestep T+1
                if t == self.param_args['len_time'] - 1:
                    break

                # Apply reduce_mask to input_data
                t_hgb_input_data_scale = t_input_data_scale[:,:,self.param_args['hgb_index']].clone()
                t_hgb_input_data_scale[(t_reduce_mask==1)&(t_stable_mask>=0)] = 0
                t_input_data_scale[:,:,self.param_args['hgb_index']] = t_hgb_input_data_scale
                
                # Apply reduce_mask to hgb_delta
                t_hgb_delta[(t_reduce_mask==1)&(t_stable_mask>=0)] = 0

                # Apply reduce_mask to nnan_mask
                t_hgb_nnan_mask = t_nnan_mask[:,:,self.param_args['hgb_index']].clone()
                t_hgb_nnan_mask[(t_reduce_mask==1)&(t_stable_mask>=0)] = 0
                t_nnan_mask[:,:,self.param_args['hgb_index']] = t_hgb_nnan_mask

                new_input_data_scale = torch.concat([new_input_data_scale,
                                                     t_input_data_scale], dim=1)
                new_hgb_delta = torch.concat([new_hgb_delta,
                                              t_hgb_delta], dim=1)
                new_nnan_mask = torch.concat([new_nnan_mask,
                                              t_nnan_mask], dim=1)

                (_, 
                 t_normal_out, t_stable_out, 
                 t_select_out) = self.model(
                    new_input_data_scale, 
                    new_hgb_delta,
                    batch_delta_time[:,:t+1],
                    batch_pfeature,
                    new_nnan_mask
                )

                normal_out.append(torch.sigmoid(t_normal_out[:,-1:]))
                stable_out.append(torch.sigmoid(t_stable_out[:,-1:]))
                select_out.append(t_select_out[:,-1:])

            normal_out = torch.concat(normal_out, dim=1)
            stable_out = torch.concat(stable_out, dim=1)
            select_out = torch.concat(select_out, dim=1)
            embed_out = torch.concat(embed_out, dim=1)
            reduce_mask = torch.concat(reduce_mask.squeeze(-1), dim=1)
            auto_mask = torch.concat(auto_mask.squeeze(-1), dim=1)
    
        select_out[batch_stable_mask[:,past_split+1:]==-1] = np.nan
        normal_out[batch_normal_mask[:,past_split+1:]==-1] = np.nan
        stable_out[batch_stable_mask[:,past_split+1:]==-1] = np.nan
        
        stable_mask = batch_stable_mask[:,past_split:].clone()
        reduce_mask = reduce_mask.float()
        reduce_mask[stable_mask[:,1:]==0] = np.nan
        auto_mask[stable_mask[:,1:]==0] = np.nan

        return (normal_out, stable_out, select_out, 
                reduce_mask, auto_mask)
    
    @torch.no_grad()
    def predict(self, test_loader, 
                reduction_mode='reduction_all', score_thres=0.5):
        self.model.eval()
        
        normal_out, stable_out = [], []
        select_out = []
        reduce_mask, auto_mask = [], []
        
        for idx, batch_data in enumerate(test_loader):            
            (batch_normal_out, batch_stable_out, batch_select_out, 
             batch_reduce_mask, batch_auto_mask) = self.predict_batch(
                batch_data, 
                reduction_mode=reduction_mode,
                score_thres=score_thres
            )
            
            normal_out.append(batch_normal_out)
            stable_out.append(batch_stable_out)
            select_out.append(batch_select_out)
            
            reduce_mask.append(batch_reduce_mask)
            auto_mask.append(batch_auto_mask)
        
        normal_out = torch.cat(normal_out)
        stable_out = torch.cat(stable_out)
        select_out = torch.cat(select_out)
        
        reduce_mask = torch.cat(reduce_mask)
        auto_mask = torch.cat(auto_mask)


        value_results_dict = {
            'Normal': normal_out.cpu(),
            'Stable': stable_out.cpu(),
            'Selections': select_out.cpu(),
            
            'Reduce mask': reduce_mask.cpu(),
            'Auto mask': auto_mask.cpu(),
        }

        return value_results_dict

