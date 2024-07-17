# +
import argparse
import config

import torch
from torchmetrics.functional import accuracy, auroc, average_precision, precision, specificity, f1_score


# -

def read_config_dict():
    # read in yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path for the config file")
    args = parser.parse_args()

    with open(cfg_filename) as file: 
        config_dict = config.Config(file)
        
    return config_dict


def get_metrics(pred, target, thres=0.5):
    metrics = {}
    
    pred_int = pred.clone().detach()
    pred_int[pred>=thres] = 1
    pred_int[pred<thres] = 0
    metrics['Acc'] = round(accuracy(pred_int.flatten(), target.long().flatten(), task='binary').item(), 4)
    metrics['Prec'] = round(precision(pred_int.flatten(), target.long().flatten(), task='binary').item(), 4)
    metrics['AUROC'] = round(auroc(pred.flatten(), target.long().flatten(), task='binary').item(), 4)
    metrics['AUPRC'] = round(average_precision(pred.flatten(), target.long().flatten(), task='binary').item(), 4)
    
    return metrics
