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
    assert label_name == 'entity' or label_name == 'action'
    
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


def get_context_len(seq):
    context_len = 0
    for line in seq[:-1]:
        context_len += len(line)
        
    return context_len


def flat_seq(label_histories, token_ids, args):
    if len(label_histories) > 1:
        seq = label_histories[:-1] + [token_ids]
        seq[0] = [args.bos_id] + seq[0]
    else:
        seq = [[args.bos_id], token_ids]
    seq[-1] = seq[-1] + [args.eos_id]
    
    context_len = get_context_len(seq)
    
    if context_len + len(seq[-1]) > args.max_len:
        return None, None

    trg_spots = (context_len+1, context_len+len(seq[-1])-1)
    seq = list(chain.from_iterable(seq))
    
    assert len(seq) <= args.max_len
    
    return pad_seq(seq, args.max_len, args.pad_id, args.eos_id), trg_spots
    

class EffERDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer, excluded=[], cached=False):
        self.target = True if args.setting.split('-')[0] == 'target' else False
        self.sep = True if args.setting.split('-')[1] == 'sep' else False

        self.input_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        if not cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='entity')  # (N, T, L), (N, T, num_entites)
            
            print(f"Processing {data_prefix} data...")
            idx = 0
            for d, dialogue in enumerate(tqdm(utters)):
                entity_histories = []
                for u, line in enumerate(dialogue):
                    speaker = line.split(':')[0]
                    if speaker == 'speaker1':
                        speaker_id = args.speaker1_id
                    else:
                        speaker_id = args.speaker2_id
                    utter = line[len(speaker)+1:].lower()
                    tokens = tokenizer.tokenize(utter)
                    
                    token_ids = [speaker_id] + [tokenizer.get_vocab()[token] for token in tokens]
                    
                    if self.target and speaker == 'speaker2':
                        entity_infos = []
                    else:
                        entity_infos = labels[d][u]
                    
                    entity_infos.sort(key=lambda x:x[2])
                    entity_seq = [f"({entity_info[0]}, {entity_info[1]})" for entity_info in entity_infos]
                    entity_seq = ' '.join(entity_seq)
                    entity_tokens = tokenizer.tokenize(entity_seq)
                    
                    speaker_list = []
                    if self.sep:
                        speaker_list.append(speaker_id)
                    
                    entity_histories.append(speaker_list + [tokenizer.get_vocab()[token] for token in entity_tokens])
                    if len(entity_histories) > args.max_times:
                        entity_histories = entity_histories[1:]

                    if speaker_id == args.speaker1_id:
                        if idx not in excluded:
                            utter_labels = ['O' for i in range(len(tokens))]
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

                            input_ids, trg_spots = flat_seq(copy.deepcopy(entity_histories), token_ids, args)
                            if input_ids is not None:
                                full_labels = [-1] * len(input_ids)
                                full_labels[trg_spots[0]:trg_spots[1]] = utter_labels

                                assert len(full_labels) == len(input_ids), f"{tokens} || {entity_tokens}"
                                
                                self.labels.append(full_labels)
                                self.input_ids.append(input_ids)
                            else:
                                exceed_count += 1
                                excluded.append(idx)
                            
                        idx += 1
            
            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
                
            with open(f"{args.ckpt_dir}/{data_prefix}_excluded.pickle", 'wb') as f:
                pickle.dump(excluded, f)
        else:
            print("Loading cached data...")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
        
        print(f"Total {len(self.input_ids)} sequences prepared.")
        
        self.input_ids = torch.LongTensor(self.input_ids)
        self.labels = torch.LongTensor(self.labels)
    
        assert args.max_len == self.input_ids.shape[1]
    
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
    

class EffAPDataset(Dataset):
    def __init__(self, args, data_prefix, class_dict, tokenizer, excluded=[], cached=False):
        self.target = True if args.setting.split('-')[0] == 'target' else False
        self.sep = True if args.setting.split('-')[1] == 'sep' else False
        
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, num_actions)
        
        if not cached:
            exceed_count = 0
            utters, action_labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='action')  # (N, T, L), (N, T, num_actions)
            _, entity_labels = load_data(args.dataset_dir, data_prefix, utter_name='utter', label_name='entity')  # (N, T, L), (N, T, num_entites)
            
            print(f"Processing {data_prefix} data...")
            idx = 0
            for d, dialogue in enumerate(tqdm(utters)):
#                 action_histories = []
                entity_histories = []
                for u, line in enumerate(dialogue):
                    speaker = line.split(':')[0]
                    if speaker == 'speaker1':
                        speaker_id = args.speaker1_id
                    else:
                        speaker_id = args.speaker2_id
                    utter = line[len(speaker)+1:]
                    tokens = tokenizer.tokenize(utter)
                    
                    token_ids = [speaker_id] + [tokenizer.get_vocab()[token] for token in tokens]
                    
#                     if self.target and speaker == 'speaker1':
#                         actions = []
                    if self.target and speaker == 'speaker2':
                        entity_infos = []
                    else:
#                         actions = labels[d][u]
                        entity_infos = entity_labels[d][u]
    
#                     action_seq = []
#                     for action in actions:
#                         if action[0] != "":
#                             action_seq.append(f"({action[0]}, {action[1]})")
#                         else:
#                             action_seq.append(f"({action[1]})")
#                     action_seq = ' '.join(action_seq)
                    
#                     action_tokens = tokenizer.tokenize(action_seq)

#                     speaker_list = []
#                     if self.sep:
#                         speaker_list.append(speaker_id)

                    entity_infos.sort(key=lambda x:x[2])
                    entity_seq = [f"({entity_info[0]}, {entity_info[1]})" for entity_info in entity_infos]
                    entity_seq = ' '.join(entity_seq)
                    entity_tokens = tokenizer.tokenize(entity_seq)
                    
                    speaker_list = []
                    if self.sep:
                        speaker_list.append(speaker_id)
                    
#                     action_histories.append(speaker_list + [tokenizer.get_vocab()[token] for token in action_tokens])
#                     if len(action_histories) > args.max_times:
#                         action_histories = action_histories[1:]

                    entity_histories.append(speaker_list + [tokenizer.get_vocab()[token] for token in entity_tokens])
                    if len(entity_histories) > args.max_times:
                        entity_histories = entity_histories[1:]

                    if speaker_id == args.speaker1_id and u < len(dialogue)-1:
                        if idx not in excluded:
                            action_ids = [class_dict[action[1]] for action in action_labels[d][u+1]]
                            target = F.one_hot(torch.LongTensor(action_ids), num_classes=args.num_classes)
                            target = (target.sum(0) > 0).long().tolist()

                            assert len(target) == len(class_dict)

#                             input_ids, _ = flat_seq(copy.deepcopy(action_histories), token_ids, args)
                            input_ids, _ = flat_seq(copy.deepcopy(entity_histories), token_ids, args)

                            if input_ids is not None:
                                self.input_ids.append(input_ids)
                                self.labels.append(target)
                            else:
                                exceed_count += 1
                                excluded.append(idx)
                            
                        idx += 1

            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
                
            with open(f"{args.ckpt_dir}/{data_prefix}_excluded.pickle", 'wb') as f:
                pickle.dump(excluded, f)
        else:
            print("Loading cached data...")
            with open(f"{args.ckpt_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.ckpt_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
        
        print(f"Total {len(self.input_ids)} sequences prepared.")
        
        self.input_ids = torch.LongTensor(self.input_ids)
        self.labels = torch.FloatTensor(self.labels)
        
        assert args.max_len == self.input_ids.shape[1]
                    
    def __len__(self):
        return self.input_ids.shape[0]
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
