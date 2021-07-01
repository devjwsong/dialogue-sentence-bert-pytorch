from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import torch
import numpy as np
import random


def pretrain_scores(preds, trues, round_num=4):
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='micro')
    
    return {'acc': round(acc, round_num), 'f1': round(f1, round_num)}


def intent_scores(preds, trues, intent_class_dict=None, round_num=4):
    # preds: list of predicted calss for each sample (N)
    # trues: list of ground-truth class for each sample (N)
    
    all_acc = accuracy_score(trues, preds)
    all_f1 = f1_score(trues, preds, average='micro')
    scores = {'all_acc': round(all_acc, round_num), 'all_f1': round(all_f1, round_num)}
    
    if intent_class_dict is not None:
        oos_idx = intent_class_dict['oos']
        oos_trues, oos_preds = [], []
        ins_trues, ins_preds = [], []
        
        for i in range(len(preds)):
            if trues[i] != oos_idx:
                ins_preds.append(preds[i])
                ins_trues.append(trues[i])

            oos_trues.append(int(trues[i] == oos_idx))
            oos_preds.append(int(preds[i] == oos_idx))
            
        ins_preds = np.array(ins_preds)
        ins_trues = np.array(ins_trues)
        oos_preds = np.array(oos_preds)
        oos_trues = np.array(oos_trues)
        ins_acc = (ins_preds == ins_trues).mean()
        oos_acc = (oos_preds == oos_trues).mean()

        # for oos samples recall = tp / (tp + fn) 
        TP = (oos_trues & oos_preds).sum()
        FN = ((oos_trues - oos_preds) > 0).sum()
        oos_recall = TP / (TP+FN)
        
        scores['ins_acc'] = round(ins_acc, round_num)
        scores['oos_acc'] = round(oos_acc, round_num)
        scores['oos_recall'] = round(oos_recall, round_num)
    
    return scores


def action_scores(preds, trues, round_num=4):
    # preds: list of predicted sublists which contains binary indicator for each label (N, num_classes)
    # trues: list of ground-truth sublists which contains binary indicator for each label (N, num_classes)
    
    samples_f1 = f1_score(trues, preds, average='samples', zero_division=0)
    micro_f1 = f1_score(trues, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    
    count = 0
    for t, true in enumerate(trues):
        if preds[t] == true:
            count += 1
    exact_acc = count / len(trues)
    
    scores = {
        'samples_f1': round(samples_f1, round_num),
        'micro_f1': round(micro_f1, round_num),
        'macro_f1': round(macro_f1, round_num),
        'exact_acc': round(exact_acc, round_num),
    }
    
    return scores
