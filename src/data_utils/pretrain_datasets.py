from torch.utils.data import Dataset
from glob import glob

import torch
import json
import natsort


class PretrainDataset(Dataset):
    def __init__(self, args):
        self.sample_files_0 = glob(f"{args.data_dir}/{args.pretrain_dir}/sample_list_0_group_*")
        self.sample_files_1 = glob(f"{args.data_dir}/{args.pretrain_dir}/sample_list_1_group_*")
        self.label_files = glob(f"{args.data_dir}/{args.pretrain_dir}/labels_group_*")
        
        self.sample_files_0 = natsort.natsorted(self.sample_files_0)
        self.sample_files_1 = natsort.natsorted(self.sample_files_1)
        self.label_files = natsort.natsorted(self.label_files)
        
        assert len(self.sample_files_0) == len(self.sample_files_1)
        assert len(self.sample_files_0) == len(self.label_files)
        
        with open(self.sample_files_0[0], 'r') as f:
            first_group = json.load(f)
        self.group_size = len(first_group)
        self.num_samples = (len(self.sample_files_0) - 1) * self.group_size
        with open(self.sample_files_0[-1], 'r') as f:
            last_group = json.load(f)
        self.num_samples += len(last_group)
        
        self.cur_group_idx = -1
        self.cur_sample_list_0, self.cur_sample_list_1, self.cur_labels = [], [], []
    
    def __getitem__(self, idx):
        group_idx = idx // self.group_size
        if group_idx != self.cur_group_idx:
            with open(self.sample_files_0[group_idx], 'r') as f:
                self.cur_sample_list_0 = json.load(f)
            with open(self.sample_files_1[group_idx], 'r') as f:
                self.cur_sample_list_1 = json.load(f)
            with open(self.label_files[group_idx], 'r') as f:
                self.cur_labels = json.load(f)
            self.cur_group_idx = group_idx
            
        sample_idx = idx % self.group_size
        return self.cur_sample_list_0[sample_idx], self.cur_sample_list_1[sample_idx], self.cur_labels[sample_idx]
    
    def __len__(self):
        return self.num_samples

    
class PretrainPadCollate():
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad_collate(self, batch):
        input_ids_0, input_ids_1, class_labels = [], [], []
        for idx, tup in enumerate(batch):
            input_ids_0.append(torch.LongTensor(tup[0]))
            input_ids_1.append(torch.LongTensor(tup[1]))
            class_labels.append(tup[2])

        padded_input_ids_0 = torch.nn.utils.rnn.pad_sequence(input_ids_0, batch_first=True, padding_value=self.input_pad_id)
        padded_input_ids_1 = torch.nn.utils.rnn.pad_sequence(input_ids_1, batch_first=True, padding_value=self.input_pad_id)
        class_labels = torch.LongTensor(class_labels)
        
        return padded_input_ids_0.contiguous(), padded_input_ids_1.contiguous(), class_labels.contiguous()
    