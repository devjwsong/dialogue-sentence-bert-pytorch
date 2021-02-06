from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm
from itertools import chain

import torch
import os
import pickle
import copy
import numpy as np


def load_data(dataset_dir, data_prefix, utter_name, label_name):
    print(f"Loading {data_prefix} pickle files...")
    with open(f"{dataset_dir}/{data_prefix}_{utter_name}.pickle", 'rb') as f:
        utters = pickle.load(f)
    with open(f"{dataset_dir}/{data_prefix}_{label_name}.pickle", 'rb') as f:
        labels = pickle.load(f)
        
    return utters, labels


def pad_seq(seq, max_len, pad_id, eos_id):
    if len(seq) <= max_len:
        seq += [pad_id] * (max_len-len(seq))
    else:
        seq = seq[:max_len]
        seq[-1] = eos_id
        
    return seq


def get_context_len(utters):
    context_len = 0
    for utter in utters[:-1]:
        context_len += len(utter)
        
    return context_len


def flat_seq(utters, args):
    if len(utters) > 1:
        utters[0] = [args.bos_id] + utters[0]
    else:
        utters.insert(0, [args.bos_id])
    utters[-1] = utters[-1] + [args.eos_id]
    
    context_len = get_context_len(utters)
    
    if context_len > args.max_len:
        return None, None
    
#     if context_len > (args.max_len-len(utters[-1])):
#         utter_len = (args.max_len-len(utters[-1])) // (args.max_times-1)
#         utters = [utter[:utter_len] for u, utter in enumerate(utters) if u != len(utters)-1]
#         context_len = get_context_len(utters)

    trg_spots = (context_len+1, context_len+len(utters[-1])-1)
    utters = list(chain.from_iterable(utters))
    
    assert len(utters) <= args.max_len
    
    return pad_seq(utters, args.max_len, args.pad_id, args.eos_id), trg_spots
    

class ERDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer):
        utters, labels = load_data(args.dataset_dir, data_prefix, args.utter_name, args.label_name)  # (N, T, L), (N, T, num_entites)
        
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        self.excluded = []
        
        print(f"Processing {data_prefix} data...")
        idx = 0
        for d, dialogue in enumerate(tqdm(utters)):
            utter_histories = []
            for u, line in enumerate(dialogue):
                speaker = line.split(':')[0]
                if speaker == 'speaker1':
                    speaker_id = args.speaker1_id
                else:
                    speaker_id = args.speaker2_id
                utter = line[len(speaker)+1:].lower()
                tokens = tokenizer.tokenize(utter)
                token_ids = [speaker_id] + [tokenizer.get_vocab()[token] for token in tokens]
                utter_histories.append(token_ids)
                if len(utter_histories) > args.max_times:
                    utter_histories = utter_histories[1:]
                
                if speaker_id == args.speaker1_id:
                    utter_labels = ['O' for i in range(len(tokens))]
                    entity_infos = labels[d][u]
                    entity_infos.sort(key=lambda x:x[2])
                    for entity_info in entity_infos:
                        entity_type = entity_info[0]
                        entity_value = entity_info[1].lower()
                        if 'gpt' in args.model_name.lower():
                            add_entity_tokens = tokenizer.tokenize(" " + entity_value)
                        else:
                            add_entity_tokens = None
                        entity_tokens = tokenizer.tokenize(entity_value)
                        utter_labels = self.update_labels(tokens, entity_type, entity_tokens, utter_labels, add_entity_tokens=add_entity_tokens)
                    
                    assert len(tokens) == len(utter_labels)
                    
                    utter_labels = [class_dict[label] for label in utter_labels]
                        
                    input_ids, trg_spots = flat_seq(copy.deepcopy(utter_histories), args)
                    if input_ids is None:
                        self.excluded.append(idx)
                    else:
                        full_labels = [-1] * len(input_ids)
                        full_labels[trg_spots[0]:trg_spots[1]] = utter_labels

                        self.labels.append(full_labels)
                        self.input_ids.append(input_ids)
                    
                    idx += 1
                    
        self.input_ids = torch.LongTensor(self.input_ids)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    
    def update_labels(self, tokens, entity_type, entity_tokens, labels, add_entity_tokens=None):
        found = False

        def find(tokens, entity_type, entity_tokens, labels, found):
            for t, token in enumerate(tokens):
                if token == entity_tokens[0] and labels[t] == 'O':
                    found = True
                    if tokens[t:t+len(entity_tokens)] == entity_tokens:
                        labels[t] = f'B-{entity_type}'
                        if len(entity_tokens) > 1:
                            labels[t+1:t+len(entity_tokens)] = [f'I-{entity_type}'] * (len(entity_tokens)-1)
                        break    
                        
            return labels, found

        labels, found = find(tokens, entity_type, entity_tokens, labels, found)

        if add_entity_tokens is not None and found is False:
            labels, found = find(tokens, entity_type, add_entity_tokens, labels, found)

        return labels
    

class APDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer):
        utters, labels = load_data(args.dataset_dir, data_prefix, args.utter_name, args.label_name)  # (N, T, L), (N, T, num_actions)
        
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, num_actions)
        
        self.excluded = []
        
        print(f"Processing {data_prefix} data...")
        idx = 0
        for d, dialogue in enumerate(tqdm(utters)):
            utter_histories = []
            for u, line in enumerate(dialogue):
                speaker = line.split(':')[0]
                if speaker == 'speaker1':
                    speaker_id = args.speaker1_id
                else:
                    speaker_id = args.speaker2_id
                utter = line[len(speaker)+1:]
                tokens = tokenizer.tokenize(utter)
                token_ids = [speaker_id] + [tokenizer.get_vocab()[token] for token in tokens]
                utter_histories.append(token_ids)
                if len(utter_histories) > args.max_times:
                    utter_histories = utter_histories[1:]
                    
                if speaker_id == args.speaker1_id:
                    actions = labels[d][u]
                    action_ids = [class_dict[action] for action in actions]
                    target = F.one_hot(torch.LongTensor(action_ids), num_classes=args.num_classes)
                    target = (target.sum(0) > 0).long().tolist()
                    
                    assert len(target) == args.class_dict
                    
                    input_id, _ = flat_seq(copy.deepcopy(utter_histories), args)
                    
                    if input_ids is None:
                        self.excluded.append(idx)
                    else:
                        self.input_ids.append(input_id)
                        self.labels.append(target)
                        
                    idx += 1
        
        assert len(self.input_ids) == len(self.labels)
        
        self.input_ids = torch.LongTensor(self.input_ids)
        self.labels = torch.FloatTensor(self.labels)
                    
    def __len__(self):
        return self.input_ids.shape[0]
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]