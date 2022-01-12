from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm
from itertools import chain

import torch
import os
import pickle
import copy


def load_data(dataset_dir, data_prefix, utter_name, label_name):
    assert label_name == 'intents' or label_name == 'actions'
    
    print(f"Loading {data_prefix} pickle files...")
    with open(f"{dataset_dir}/{data_prefix}_{utter_name}.pickle", 'rb') as f:
        utters = pickle.load(f)
    with open(f"{dataset_dir}/{data_prefix}_{label_name}.pickle", 'rb') as f:
        labels = pickle.load(f)
        
    return utters, labels


def make_single_turn_seq(token_ids, args):
    seq = [args.cls_id] + token_ids + [args.sep_id]
    if len(seq) > args.max_encoder_len:
        seq = seq[:args.max_encoder_len]
        seq[-1] = args.sep_id
        
    return seq

def flat_seq(utter_hists, args):
    prev_hists = utter_hists[:-1]
    input_ids = None
    for i in range(len(prev_hists)):
        context = utter_hists[i:]
        context = list(chain.from_iterable(context))
        seq_len = len(context) + len(utter_hists[-1]) + 3

        if seq_len <= args.max_encoder_len:
            input_ids = [args.cls_id] + context + [args.sep_id] + utter_hists[-1] + [args.sep_id]
            context_len = len(context) + 2
            query_len = len(input_ids) - context_len
            
            break

    if input_ids is None:
        input_ids = make_single_turn_seq(utter_hists[-1], args)
            
    return input_ids


class IDDataset(Dataset):
    def __init__(self, args, data_prefix, tokenizer):
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, L)
        
        self.data_prefix = data_prefix
        
        max_len = 0
        if not args.cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utters', label_name='intents')  # (N, L), (N)
            
            print(f"Processing {data_prefix} data...")
            for u, utter in enumerate(tqdm(utters)):
                tokens = tokenizer.tokenize(utter)
                token_ids = [args.cls_id] + tokenizer.convert_tokens_to_ids(tokens) + [args.sep_id]

                if len(token_ids) > args.max_encoder_len:
                    exceed_count += 1
                    token_ids = token_ids[:args.max_encoder_len]
                    token_ids[-1] = args.sep_id
                
                max_len = max(max_len, len(token_ids))
                self.input_ids.append(token_ids)
                self.labels.append(args.class_dict[labels[u]])
                    
            assert len(self.input_ids) == len(self.labels)
            
            print(f"Exceed count: {exceed_count}")
            assert exceed_count == 0
            print(f"Max length: {max_len}")
            with open(f"{args.cached_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.cached_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            print("Loading cached data...")
            with open(f"{args.cached_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.cached_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
                
        print(f"Total {len(self.input_ids)} sequences prepared.")
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    

class APDataset(Dataset):
    def __init__(self, args, data_prefix, tokenizer):
        self.input_ids = []  # (N, L)
        self.labels = []  # (N, num_actions)
        
        max_len = 0
        if not args.cached:
            exceed_count = 0
            utters, labels = load_data(args.dataset_dir, data_prefix, utter_name='utters', label_name='actions')  # (N, T), (N, T, num_actions)
            
            print(f"Processing {data_prefix} data...")
            for d, dialogue in enumerate(tqdm(utters)):
                utter_hists = []
                for u, line in enumerate(dialogue):
                    speaker = line.split(':')[0]
                    utter = line[len(speaker)+1:]
                    tokens = tokenizer.tokenize(utter)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    utter_hists.append(token_ids)
                    
                    if len(utter_hists) > args.max_turns:
                        utter_hists = utter_hists[1:]

                    if speaker == 'usr' and u < len(dialogue)-1:
                        actions = labels[d][u+1]
                        action_ids = [args.class_dict[action[1]] for action in actions]
                        target = F.one_hot(torch.LongTensor(action_ids), num_classes=args.num_classes)
                        target = (target.sum(0) > 0).long().tolist()

                        assert len(target) == len(args.class_dict)

                        input_ids = flat_seq(copy.deepcopy(utter_hists), args)
                        self.input_ids.append(input_ids)
                        self.labels.append(target)
                        max_len = max(max_len, len(input_ids))

            assert len(self.input_ids) == len(self.labels)
            
            print(f"Max length: {max_len}")
            with open(f"{args.cached_dir}/{data_prefix}_input_ids_cached.pickle", 'wb') as f:
                pickle.dump(self.input_ids, f)
            with open(f"{args.cached_dir}/{data_prefix}_labels_cached.pickle", 'wb') as f:
                pickle.dump(self.labels, f)
        else:
            print("Loading cached data...")
            with open(f"{args.cached_dir}/{data_prefix}_input_ids_cached.pickle", 'rb') as f:
                self.input_ids = pickle.load(f)
            with open(f"{args.cached_dir}/{data_prefix}_labels_cached.pickle", 'rb') as f:
                self.labels = pickle.load(f)
        
        print(f"Total {len(self.input_ids)} sequences prepared.")
                    
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    

class PadCollate():
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
