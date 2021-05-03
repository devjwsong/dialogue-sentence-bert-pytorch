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
    assert label_name == 'intent' == label_name == 'entity' or label_name == 'action'
    
    print(f"Loading {data_prefix} pickle files...")
    with open(f"{dataset_dir}/{data_prefix}_{utter_name}.pickle", 'rb') as f:
        utters = pickle.load(f)
    with open(f"{dataset_dir}/{data_prefix}_{label_name}.pickle", 'rb') as f:
        labels = pickle.load(f)
        
    return utters, labels


def get_context_len(utters):
    context_len = 0
    for utter in utters[:-1]:
        context_len += len(utter)
        
    return context_len


def flat_seq(utters, args):
    if len(utters) > 1:
        utters[0] = [args.cls_id] + utters[0]
    else:
        utters.insert(0, [args.cls_id])
    utters[-1] = [args.sep_id] + utters[-1] + [args.sep_id]
    
    context_len = get_context_len(utters)
    
    if context_len + len(utters[-1]) > args.max_encoder_len:
        return None, None

    trg_spots = (context_len+1, context_len+len(utters[-1])-1)
    utters = list(chain.from_iterable(utters))
    
    return utters, trg_spots


class IDDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer, cached=False):
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        max_len = 0
        
        if not cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='intent')  # (N, L), (N)
            
            print(f"Processing {data_prefix} data...")
            for u, utter in enumerate(tqdm(utters)):
                tokens = tokenizer.tokenize(utter)
                token_ids = [args.cls_id] + tokenizer.convert_tokens_to_ids(tokens) + [args.sep_id]
                
                if len(token_ids) > args.max_encoder_len:
                    exceed_count += 1
                else:
                    max_len = max(max_len, len(token_ids))
                    self.input_ids.append(token_ids)
                    self.labels.append(class_dict[labels[u]])
                    
            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            print(f"Max length: {max_len}")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)   
        else:
            print("Loading cached data...")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
                
        print(f"Total {len(self.input_ids)} sequences prepared.")
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


class ERDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer, cached=False):
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        max_len = 0
        
        if not cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='entity')  # (N, T), (N, T, num_entites)
            
            print(f"Processing {data_prefix} data...")
            for d, dialogue in enumerate(tqdm(utters)):
                utter_histories = []
                for u, line in enumerate(dialogue):
                    speaker = line.split(':')[0]
                    utter = line[len(speaker)+1:].lower()
                    tokens = tokenizer.tokenize(utter)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    utter_histories.append(token_ids)
                    if len(utter_histories) > args.max_turns:
                        utter_histories = utter_histories[1:]

                    if speaker == 'usr':
                        utter_labels = ['O' for i in range(len(tokens))]
                        entity_infos = labels[d][u]
                        entity_infos.sort(key=lambda x:x[2])
                        for entity_info in entity_infos:
                            entity_type = entity_info[0]
                            entity_value = entity_info[1].lower()

                            entity_tokens = tokenizer.tokenize(entity_value)
                            utter_labels = self.update_labels(tokens, entity_type, entity_tokens, utter_labels)

                        assert len(tokens) == len(utter_labels)

                        utter_labels = [class_dict[label] for label in utter_labels]

                        input_ids, trg_spots = flat_seq(copy.deepcopy(utter_histories), args)
                        if input_ids is not None:
                            full_labels = [-1] * len(input_ids)
                            full_labels[trg_spots[0]:trg_spots[1]] = utter_labels

                            self.labels.append(full_labels)
                            self.input_ids.append(input_ids)
                            
                            max_len = max(max_len, len(input_ids))
                        else:
                            exceed_count += 1
            
            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            print(f"Max length: {max_len}")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            print("Loading cached data...")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
        
        print(f"Total {len(self.input_ids)} sequences prepared.")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    
    def update_labels(self, tokens, entity_type, entity_tokens, labels):
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

        return labels
    

class APDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer, cached=False):
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, num_actions)
        
        max_len = 0
        
        if not cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='action')  # (N, T), (N, T, num_actions)
            
            print(f"Processing {data_prefix} data...")
            for d, dialogue in enumerate(tqdm(utters)):
                utter_histories = []
                for u, line in enumerate(dialogue):
                    speaker = line.split(':')[0]
                    utter = line[len(speaker)+1:]
                    tokens = tokenizer.tokenize(utter)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    utter_histories.append(token_ids)
                    if len(utter_histories) > args.max_turns:
                        utter_histories = utter_histories[1:]

                    if speaker == 'usr' and u < len(dialogue)-1:
                        actions = labels[d][u+1]
                        action_ids = [class_dict[action[1]] for action in actions]
                        target = F.one_hot(torch.LongTensor(action_ids), num_classes=args.num_classes)
                        target = (target.sum(0) > 0).long().tolist()

                        assert len(target) == len(class_dict)

                        input_ids, _ = flat_seq(copy.deepcopy(utter_histories), args)

                        if input_ids is not None:
                            self.input_ids.append(input_ids)
                            self.labels.append(target)
                            
                            max_len = max(max_len, len(input_ids))
                        else:
                            exceed_count += 1

            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            print(f"Max length: {max_len}")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            print("Loading cached data...")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
        
        print(f"Total {len(self.input_ids)} sequences prepared.")
                    
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    

class IntentPadCollate():
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad_collate(self, batch):
        input_ids, labels = [], []
        for idx, pair in enumerate(batch):
            input_ids.append(torch.LongTensor(pair[0]))
            labels.append(pair[1])

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.input_pad_id)
        labels = torch.LongTensor(labels)
        
        return padded_input_ids.contiguous(), labels.contiguous()
    
    
class EntityPadCollate():
    def __init__(self, input_pad_id, label_pad_id):
        self.input_pad_id = input_pad_id
        self.label_pad_id = label_pad_id

    def pad_collate(self, batch):
        input_ids, labels = [], []
        for idx, pair in enumerate(batch):
            input_ids.append(torch.LongTensor(pair[0]))
            labels.append(torch.LongTensor(pair[1]))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.input_pad_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_id)
        
        return padded_input_ids.contiguous(), padded_labels.contiguous()

    
class ActionPadCollate():
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad_collate(self, batch):
        input_ids, labels = [], []
        for idx, pair in enumerate(batch):
            input_ids.append(torch.LongTensor(pair[0]))
            labels.append(pair[1])

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.input_pad_id)
        labels = torch.LongTensor(labels)
        
        return padded_input_ids.contiguous(), labels.contiguous()
