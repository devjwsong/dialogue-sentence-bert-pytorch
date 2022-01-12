from torch.utils.data import Dataset
from glob import glob

import torch
import ujson
import natsort


class PretrainDataset(Dataset):
    def __init__(self, args):
        self.first_sample_files = glob(f"{args.shuffled_dir}/first_samples_group*")
        self.second_sample_files = glob(f"{args.shuffled_dir}/second_samples_group*")
        self.label_files = glob(f"{args.shuffled_dir}/label_group*")
        
        self.first_sample_files = natsort.natsorted(self.first_sample_files)
        self.second_sample_files = natsort.natsorted(self.second_sample_files)
        self.label_files = natsort.natsorted(self.label_files)
        
        assert len(self.first_sample_files) == len(self.second_sample_files)
        assert len(self.first_sample_files) == len(self.label_files)
        
        with open(self.first_sample_files[0], 'r') as f:
            first_group = ujson.load(f)
        self.group_size = len(first_group)
        self.num_samples = (len(self.first_sample_files) - 1) * self.group_size
        with open(self.first_sample_files[-1], 'r') as f:
            last_group = ujson.load(f)
        self.num_samples += len(last_group)
        
        self.cur_group_idx = -1
        self.cur_first_samples, self.cur_second_samples, self.cur_labels = [], [], []
    
    def __getitem__(self, idx):
        group_idx = idx // self.group_size
        if group_idx != self.cur_group_idx:
            with open(self.first_sample_files[group_idx], 'r') as f:
                self.cur_first_samples = ujson.load(f)
            with open(self.second_sample_files[group_idx], 'r') as f:
                self.cur_second_samples = ujson.load(f)
            with open(self.label_files[group_idx], 'r') as f:
                self.cur_labels = ujson.load(f)
            self.cur_group_idx = group_idx
            
        sample_idx = idx % self.group_size
        return self.cur_first_samples[sample_idx], self.cur_second_samples[sample_idx], self.cur_labels[sample_idx]
    
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
    