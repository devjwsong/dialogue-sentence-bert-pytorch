from sklearn.metrics import accuracy_score as sklearn_acc
from sklearn.metrics import f1_score as sklearn_f1
from seqeval.metrics import f1_score as seqeval_f1
from itertools import chain

import torch
import numpy as np

# Entity Recognition: Sequence-level accuracy, Entity-level micro-f1, Entity-level macro-f1
# Action Prediction: mirco-F1, macro-F1


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
