from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import os
import pickle
import copy
import natsort
import random
import numpy as np


def load_data(data_list, pretrain_dir, data_prefix):
    utters, actions = [], []
    
    print(f"Loading {data_prefix} pickle files...")
    for data in data_list:
        with open(f"{pretrain_dir}/{data}/{data_prefix}_utters.pickle", 'rb') as f:
            utters += pickle.load(f)
        with open(f"{pretrain_dir}/{data}/{data_prefix}_actions.pickle", 'rb') as f:
            actions += pickle.load(f)
        
    return utters, actions


def get_context_len(utters):
    context_len = 0
    for utter in utters[:-1]:
        context_len += len(utter)
        
    return context_len


def flat_seq(utters, args):
    if len(utters) > 1:
        utters[0] = [args.cls_id] + utters[0]
        utters[-2] = utters[-2] + [args.sep_id]
    else:
        utters.insert(0, [args.cls_id])
    utters[-1] = utters[-1] + [args.sep_id]
    
    context_len = get_context_len(utters)
    if context_len + len(utters[-1]) > args.max_encoder_len:
        return None

    seq = list(chain.from_iterable(utters))
    
    return seq

def mask_seq(seq, args):
    num_masks = max(1, int((len(seq)-2) * args.masking_ratio))
    lm_label = [-1] * len(seq)
    mask_idxs = random.sample(list(range(1, len(seq)-1)), num_masks)
    for idx in mask_idxs:
        lm_label[idx] = seq[idx]
        seq[idx] = args.mask_id
    
    assert len(seq) == len(lm_label)
    
    return seq, lm_label


def make_action2seqs(args, utters, actions, tokenizer):
    action2seqs = {}
    
    for d, dialogue in enumerate(tqdm(utters)):
        utter_hists = []
        for u, utter in enumerate(dialogue):
            tokens = tokenizer.tokenize(utter)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            utter_hists.append(token_ids)
            
            if len(utter_hists) > args.max_turns:
                utter_hists = utter_hists[1:]
            
            action_list = actions[d][u]
            if len(action_list) > 0:
                seq = flat_seq(copy.deepcopy(utter_hists), args)
                if seq is not None:
                    action_set = frozenset(sorted(action_list))
                    if action_set not in action2seqs:
                        action2seqs[action_set] = []
                    action2seqs[action_set].append(seq)
                        
    return action2seqs


def save_pickles(cached_dir, data_prefix, input_prefix, label_prefix, cur_group_idx, input_ids_0, input_ids_1, class_labels, lm_labels):
    assert len(input_ids_0) == len(input_ids_1)
    assert len(input_ids_0) == len(class_labels)
    assert len(input_ids_0) == len(lm_labels)

    with open(f"{cached_dir}/{data_prefix}_{input_prefix}_0_group{cur_group_idx}", 'wb') as f:
        pickle.dump(input_ids_0, f)
    with open(f"{cached_dir}/{data_prefix}_{input_prefix}_1_group{cur_group_idx}", 'wb') as f:
        pickle.dump(input_ids_1, f)
    with open(f"{cached_dir}/{data_prefix}_class_{label_prefix}_group{cur_group_idx}", 'wb') as f:
        pickle.dump(class_labels, f)
    with open(f"{cached_dir}/{data_prefix}_lm_{label_prefix}_group{cur_group_idx}", 'wb') as f:
        pickle.dump(lm_labels, f)

        
def is_finished(a0, a1, i, j, num_actions, num_seqs_0, num_seqs_1):
    return i == num_seqs_0-1 and j == num_seqs_1-1 and a0 == num_actions-2 and a1 == num_actions-1


class PretrainDataset(Dataset):
    def __init__(self, args, data_prefix, tokenizer, class_dict):
        self.group_size = args.group_size
        self.cached_dir = args.cached_dir
        self.data_prefix = data_prefix
        self.input_prefix = args.input_prefix
        self.label_prefix = args.label_prefix
        
        if self.data_prefix == args.train_prefix:
            num_same_samples = args.num_train_same_samples
            num_diff_samples = args.num_train_diff_samples
            num_neut_samples = args.num_train_neut_samples
        elif self.data_prefix == args.valid_prefix:
            num_same_samples = args.num_valid_same_samples
            num_diff_samples = args.num_valid_diff_samples
            num_neut_samples = args.num_valid_neut_samples
        
        if not args.cached:
            print("Since you chosed not to use cached data, pre-processing is conducted first.")
            max_len = 0
            num_seqs = {v: 0 for k, v in class_dict.items()}

            data_list = os.listdir(args.pretrain_dir)
            data_list = [data for data in data_list if not data.startswith('.')]
            utters, actions = load_data(data_list, args.pretrain_dir, data_prefix)
            
            print("Making action-seq distributions...")
            action2seqs = make_action2seqs(args, utters, actions, tokenizer)
            for action, seqs in action2seqs.items():
                print(f"{action}: {len(seqs)}")
                
            print("Sampling pairs...")
            random.seed(args.seed)
            input_ids_0, input_ids_1, class_labels, lm_labels = [], [], [], []
            cur_group_idx = 0
            for a0, (action0, seqs0) in enumerate(action2seqs.items()):
                for a1, (action1, seqs1) in enumerate(action2seqs.items()):
                    random.shuffle(seqs0)
                    random.shuffle(seqs1)
                    if a0 == a1:
                        print(f"Same action: {action0} : {action1}")
                        sampled_seqs0, sampled_seqs1 = seqs0[:num_same_samples], seqs1[num_same_samples:2*num_same_samples]
                        for s0, seq0 in enumerate(tqdm(sampled_seqs0)):
                            for s1, seq1 in enumerate(sampled_seqs1):
                                if s0 < s1:
                                    max_len = max(max_len, max(len(seq0), len(seq1)))
                                    mask_seq_idx = random.sample([0,1], 1)
                                    if mask_seq_idx == 0:
                                        masked_seq_0, lm_label = mask_seq(copy.deepcopy(seq0), args)
                                        input_ids_0.append(masked_seq_0)
                                        input_ids_1.append(seq1)
                                    else:
                                        masked_seq_1, lm_label = mask_seq(copy.deepcopy(seq1), args)
                                        input_ids_0.append(seq0)
                                        input_ids_1.append(masked_seq_1)
                                    class_labels.append(class_dict['same'])
                                    lm_labels.append(lm_label)
                                    num_seqs[class_dict['same']] += 1
                                    
                                    if len(input_ids_0) == args.group_size \
                                            or is_finished(a0, a1, s0, s1, len(action2seqs), len(sampled_seqs0), len(sampled_seqs1)):
                                        save_pickles(
                                            self.cached_dir, self.data_prefix, self.input_prefix, self.label_prefix, 
                                            cur_group_idx, input_ids_0, input_ids_1, class_labels, lm_labels
                                        )
                                        input_ids_0, input_ids_1, class_labels, lm_labels = [], [], [], []
                                        cur_group_idx += 1
                    elif a0 < a1:
                        if ((action0 - action1) == action0):
                            print(f"Different action: {action0} : {action1}")
                            sampled_seqs0, sampled_seqs1 = seqs0[:num_diff_samples], seqs1[:num_diff_samples]
                            class_name = 'diff'
                        elif len(action0 & action1) > 0:
                            print(f"Neutral action: {action0} : {action1}")
                            sampled_seqs0, sampled_seqs1 = seqs0[:num_neut_samples], seqs1[:num_neut_samples]
                            class_name = 'neut'
                        else:
                            continue
                        
                        for s0, seq0 in enumerate(tqdm(sampled_seqs0)):
                            for s1, seq1 in enumerate(sampled_seqs1):
                                max_len = max(max_len, max(len(seq0), len(seq1)))
                                mask_seq_idx = random.sample([0,1], 1)
                                if mask_seq_idx == 0:
                                    masked_seq_0, lm_label = mask_seq(copy.deepcopy(seq0), args)
                                    input_ids_0.append(masked_seq_0)
                                    input_ids_1.append(seq1)
                                else:
                                    masked_seq_1, lm_label = mask_seq(copy.deepcopy(seq1), args)
                                    input_ids_0.append(seq0)
                                    input_ids_1.append(masked_seq_1)
                                class_labels.append(class_dict['same'])
                                lm_labels.append(lm_label)
                                num_seqs[class_dict[class_name]] += 1

                                if len(input_ids_0) == args.group_size \
                                        or is_finished(a0, a1, s0, s1, len(action2seqs), len(sampled_seqs0), len(sampled_seqs1)):
                                    save_pickles(
                                        self.cached_dir, self.data_prefix, self.input_prefix, self.label_prefix, 
                                        cur_group_idx, input_ids_0, input_ids_1, class_labels, lm_labels
                                    )
                                    input_ids_0, input_ids_1, class_labels, lm_labels = [], [], [], []
                                    cur_group_idx += 1
            
            print(f"<Data spec for {data_prefix} dataset>")
            print("# of inputs")
            print(num_seqs)
            print(f"Max length: {max_len}")
            

        # Load file list
        input_list_0 = [file_name for file_name in os.listdir(f"{self.cached_dir}") if file_name.startswith(f"{self.data_prefix}_{self.input_prefix}_0")]
        input_list_1 = [file_name for file_name in os.listdir(f"{self.cached_dir}") if file_name.startswith(f"{self.data_prefix}_{self.input_prefix}_1")]
        class_label_list = [file_name for file_name in os.listdir(f"{self.cached_dir}") if file_name.startswith(f"{self.data_prefix}_class_{self.label_prefix}")]
        lm_label_list = [file_name for file_name in os.listdir(f"{self.cached_dir}") if file_name.startswith(f"{self.data_prefix}_lm_{self.label_prefix}")]
        
        self.input_list_0 = natsort.natsorted(input_list_0)
        self.input_list_1 = natsort.natsorted(input_list_1)
        self.class_label_list = natsort.natsorted(class_label_list)
        self.lm_label_list = natsort.natsorted(lm_label_list)
        
        self.total_num = (len(self.input_list_0)-1) * self.group_size
        with open(f"{self.cached_dir}/{self.input_list_0[-1]}", 'rb') as f:
            seqs = pickle.load(f)
            self.total_num += len(seqs)
            
        self.cur_group_idx = -1;
        self.cur_input_ids_0, self.cur_input_ids_1, self.class_labels, self.lm_labels = [], [], [], []
        
    def __len__(self):
        return self.total_num
    
    def __getitem__(self, idx):
        group_idx = idx // self.group_size
        
        if self.cur_group_idx != group_idx:
            with open(f"{self.cached_dir}/{self.input_list_0[group_idx]}", 'rb') as f:
                self.input_ids_0 = pickle.load(f)
            with open(f"{self.cached_dir}/{self.input_list_1[group_idx]}", 'rb') as f:
                self.input_ids_1 = pickle.load(f)
            with open(f"{self.cached_dir}/{self.class_label_list[group_idx]}", 'rb') as f:
                self.class_labels = pickle.load(f)
            with open(f"{self.cached_dir}/{self.lm_label_list[group_idx]}", 'rb') as f:
                self.lm_labels = pickle.load(f)
        
        seq_idx = idx % self.group_size

        return self.input_ids_0[seq_idx], self.input_ids_1[seq_idx], self.class_labels[seq_idx], self.lm_labels[seq_idx]

    
class PretrainPadCollate():
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad_collate(self, batch):
        input_ids_0, input_ids_1, class_labels, lm_labels = [], [], [], []
        for idx, tup in enumerate(batch):
            input_ids_0.append(torch.LongTensor(tup[0]))
            input_ids_1.append(torch.LongTensor(tup[1]))
            class_labels.append(tup[2])
            lm_labels.append(torch.LongTensor(tup[3]))

        padded_input_ids_0 = torch.nn.utils.rnn.pad_sequence(input_ids_0, batch_first=True, padding_value=self.input_pad_id)
        padded_input_ids_1 = torch.nn.utils.rnn.pad_sequence(input_ids_1, batch_first=True, padding_value=self.input_pad_id)
        class_labels = torch.LongTensor(class_labels)
        padded_lm_labels = torch.nn.utils.rnn.pad_sequence(lm_labels, batch_first=True, padding_value=-1)
        
        return padded_input_ids_0.contiguous(), padded_input_ids_1.contiguous(), class_labels.contiguous(), padded_lm_labels.contiguous()
    