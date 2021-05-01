from sklearn.metrics import accuracy_score as sklearn_acc
from sklearn.metrics import f1_score as sklearn_f1
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score, recall_score
from itertools import chain

import torch
import numpy as np
import random


def entity_scores(preds, trues, class_dict, round_num=4):
    # preds: list of predicted sublists at last sequence length which contains each label for entity (N, L)
    # trues: list of ground-truth sublists at last sequence length which contains each label for entity (N, L)
    
    index_dict = {}
    for entity, idx in class_dict.items():
        index_dict[idx] = entity
    
    text_trues = [[index_dict[idx] for idx in true] for true in trues]
    text_preds = [[index_dict[idx] for idx in pred] for pred in preds]
    
    entity_micro_f1 = seqeval_f1(text_trues, text_preds, average='micro')
    entity_macro_f1 = seqeval_f1(text_trues, text_preds, average='macro')
    entity_micro_precision = precision_score(text_trues, text_preds, average='micro')
    entity_micro_recall = recall_score(text_trues, text_preds, average='micro')
    
    count = 0
    for t, true in enumerate(trues):
        if preds[t] == true:
            count += 1
    exact_acc = count / len(trues)
    
    scores = {
        'entity_micro_f1': round(entity_micro_f1, round_num),
        'entity_macro_f1': round(entity_macro_f1, round_num),
        'entity_micro_precision': round(entity_micro_precision, round_num),
        'entity_micro_recall': round(entity_micro_recall, round_num),
        'exact_acc': round(exact_acc, round_num)
    }
    
    return scores


def action_scores(preds, trues, round_num=4):
    # preds: list of predicted sublists which contains binary indicator for each label (N, num_classes)
    # trues: list of ground-truth sublists which contains binary indicator for each label (N, num_classes)
    
    micro_f1 = sklearn_f1(trues, preds, average='micro', zero_division=0)
    macro_f1 = sklearn_f1(trues, preds, average='macro', zero_division=0)
    samples_f1 = sklearn_f1(trues, preds, average='samples', zero_division=0)
    
    count = 0
    for t, true in enumerate(trues):
        if preds[t] == true:
            count += 1
    exact_acc = count / len(trues)
    
    scores = {
        'micro_f1': round(micro_f1, round_num),
        'macro_f1': round(macro_f1, round_num),
        'samples_f1': round(samples_f1, round_num),
        'exact_acc': round(exact_acc, round_num)
    }
    
    return scores


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
