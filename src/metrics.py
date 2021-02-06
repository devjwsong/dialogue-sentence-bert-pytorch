from sklearn.metrics import accuracy_score as sklearn_acc
from sklearn.metrics import f1_score as sklearn_f1
from seqeval.metrics import f1_score as seqeval_f1
from itertools import chain

import torch
import numpy as np

# Intent Detection: Acc(all), Acc(in), Acc(out), Recall(out)
# Entity Recognition: Sequence-level accuracy, Entity-level micro-f1, Entity-level macro-f1
# Dialog State Tracking: Joint accuracy, Slot accuracy
# Action Prediction: mirco-F1, macro-F1


def intent_scores(preds, trues, out_id=None):
    # preds: list of all predicted values (N)
    # trues: list of all ground truths (N)
    
    all_acc = sklearn_acc(trues, preds)
    if out_id is not None:
        in_preds = []
        in_trues = []
        out_preds = []
        out_trues = []
        for t, true in enumerate(trues):
            if true != out_id:
                in_preds.append(preds[t])
                in_trues.append(true)
            out_preds.append(int(preds[t] == out_id))
            out_trues.append(int(true == out_id))
            
        in_trues = np.array(in_trues)
        in_preds = np.array(in_preds)
        out_trues = np.array(out_trues)
        out_preds = np.array(out_preds)
                
        in_acc = (in_trues == in_preds).mean()
        out_acc = (out_trues == out_preds).mean()
        
        TP = (out_trues & out_preds).sum()
        FN = ((out_trues - out_preds) > 0).sum()
        out_recall = TP / (TP+FN)
        
        scores = {
            'all_acc': all_acc,
            'in_acc': in_acc,
            'out_acc': out_acc,
            'out_recall': out_recall
        }
        
    else:
        micro_f1 = sklearn_f1(trues, preds, average='micro')
        macro_f1 = sklearn_f1(trues, preds, average='macro')
        
        scores = {
            'all_acc': all_acc,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1
        }
    
    return scores

def entity_scores(preds, trues, class_dict):
    # preds: list of predicted sublists at last sequence length which contains each label for entity (N, L)
    # trues: list of ground-truth sublists at last sequence length which contains each label for entity (N, L)
    
    index_dict = {}
    for entity, idx in class_dict.items():
        index_dict[idx] = entity
    
    text_trues = [[index_dict[idx] for idx in true] for true in trues]
    text_preds = [[index_dict[idx] for idx in pred] for pred in preds]
    
    entity_micro_f1 = seqeval_f1(text_trues, text_preds, average='micro')
    entity_macro_f1 = seqeval_f1(text_trues, text_preds, average='macro')
    
    count = 0
    for t, true in enumerate(trues):
        if preds[t] == true:
            count += 1
    exact_acc = count / len(trues)
    
    scores = {
        'entity_micro_f1': entity_micro_f1,
        'entity_macro_f1': entity_macro_f1,
        'exact_acc': exact_acc
    }
    
    return scores
    

# def state_scores(preds, trues):
#     # preds: list of predicted sublists which contains each label id for each slot type (N, num_slots)
#     # trues: list of ground-truth sublists which contains each label id for each slot type (N, num_slots)
    
#     count = 0
#     for t, true in enumerate(trues):
#         if preds[t] == true:
#             count += 1
#     joint_acc = count / len(trues)
    
#     flat_preds = list(chain.from_iterable(preds))
#     flat_trues = list(chain.from_iterable(trues))
#     slot_acc = sklearn_acc(flat_trues, flat_preds)
    
#     return joint_acc, slot_acc


def action_scores(preds, trues):
    # preds: list of predicted sublists which contains binary indicator for each label (N, num_classes)
    # trues: list of ground-truth sublists which contains binary indicator for each label (N, num_classes)
    
    micro_f1 = sklearn_f1(trues, preds, average='micro')
    macro_f1 = sklearn_f1(trues, preds, average='macro')
    
    count = 0
    for t, true in enumerate(trues):
        if preds[t] == true:
            count += 1
    exact_acc = count / len(trues)
    
    scores = {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'exact_acc': exact_acc
    }
    
    return scores
