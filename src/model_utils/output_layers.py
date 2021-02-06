from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import torch


def load_finetune_layer(args, class_dict=None, pretrained_model=None, tokenizer=None):
    if args.task == 'intent detection':
        args.num_classes = len(class_dict)

        id_layer = IntentDetection(args)
        id_layer.init_params()

        return id_layer, args

#         elif args.task == 'dialog state tracking':
#             args.num_slots = len(class_dict)
#             args.num_labels = []

#             for slot_type, v in class_dict.items():
#                 args.num_labels.append(len(v[1]))

#             dst_layer = DialogStateTracking(args)
#             dst_layer.init_params(args, pretrained_model=pretrained_model, class_dict=class_dict, tokenizer=tokenizer)

#             return dst_layer, args

    elif args.task == 'entity recognition':
        args.num_classes = len(class_dict)

        er_layer = EntityRecognition(args)
        er_layer.init_params()

        return er_layer, args

    elif args.task == 'action prediction':
        args.num_classes = len(class_dict)

        ap_layer = ActionPrediction(args)
        ap_layer.init_params()

        return ap_layer, args
        

class LanguageModeling(nn.Module):
    def __init__(self, args):
        super(LanguageModeling, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, hiddens):
        # trg_hiddens: (B, L, d_h)
        
        return self.linear(trg_hiddens)  # (B, L, V)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
        
class IntentDetection(nn.Module):
    def __init__(self, args):
        super(IntentDetection, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, trg_hiddens):
        # trg_hiddens: (B, d_h) ([CLS] or mean)
        
        return self.linear(trg_hiddens)  # (B, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
        
class EntityRecognition(nn.Module):
    def __init__(self, args):
        super(EntityRecognition, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, hiddens):
        # hiddens: (B, L, d_h)
        
        return self.linear(hiddens)  # (B, L, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
    
    
# class DialogStateTracking(nn.Module):
#     def __init__(self, args):
#         super(DialogStateTracking, self).__init__()
        
#         self.hidden_size = args.hidden_size
#         self.num_slots = args.num_slots
#         self.num_labels = args.num_labels  # List (num_slots)
        
#         assert self.num_slots == len(self.num_labels)
        
#         self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.hidden_size) for num_label in self.num_labels])  # (num_slots, num_labels, d_h)
        
#         self.w1 = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_slots)])
#         self.w2 = nn.ModuleList([nn.Linear(2*self.hidden_size, self.hidden_size) for _ in range(self.num_slots)])
#         self.w3 = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(self.num_slots)])

#     def forward(self, trg_hiddens):
#         # trg_hiddens: (B, d_h) ([CLS] or [EOS])
#         batch_size = trg_hiddens.shape[0]
        
#         preds = []  # List (num_slots, B, num_labels)
#         for s in range(self.num_slots):
#             value_vecs = self.value_lookup[s].weight  # (num_labels, d_h)
#             num_labels = value_vecs.shape[0]
            
#             hidden = F.gelu(self.w1[s](trg_hidden_state))  # (B, d_h)
#             hidden = torch.cat([value_vecs.unsqueeze(0).repeat(batch_size, 1, 1), hidden_unsqueeze(1).repeat(1, num_labels, 1)], dim=2)  # (B, num_labels, 2d_h)
#             hidden = F.gelu(self.w2[s](hidden))  # (B, num_labels, d_h)
#             hidden = self.w3[s](hidden)  # (B, num_labels, 1)
#             dists = hidden.squeeze(-1)  # (B, num_labels)
            
#             preds.append(pred)
            
#         return preds
        
#     def init_params(self, args, pretrained_model, class_dict, tokenizer):
#         pretrained_model = pretrained_model.to(args.device)
#         pretrained_model.eval()
#         for p in pretrained_model.parameters():
#             p.requires_grad = False
            
#         pad_id = tokenizer.get_vocab()[args.pad_token]
        
#         for slot_type, v in tqdm(class_dict.items()):
#             values_tokenized = []  # (num_labels, S_L)
#             trg_poses = []  # (num_labels)
#             for value, value_id in v[1].items():
#                 start = args.bos_token
#                 end = args.eos_token
#                 value_tokens = [start, args.speaker2_token] + tokenizer.tokenize(value) + [end]
#                 if len(value_tokens) <= args.max_len:
#                     value_tokens += [args.pad_token] * (args.max_len - len(value_tokens))
#                 else:
#                     value_tokens = value_tokens[:args.max_len]
#                     value_tokens[-1] = end
                    
#                 value_token_ids = tokenizer.convert_tokens_to_ids(value_tokens)  # (S_L)
#                 values_tokenized.append(value_token_ids)
            
#             value_ids = torch.LongTensor(values_tokenized).to(args.device)  # (num_labels, S_L)
#             trg_idxs = torch.LongTensor(trg_poses).to(args.device)  # (num_labels)
#             padding_masks = (value_ids != pad_id).float().to(args.device) # (num_labels, S_L)
            
#             outputs = pretrained_model(value_ids, attention_mask=padding_masks)[0]  # (num_labels, S_L, d_h)
#             if 'bert' in args.model_name.lower():
#                 value_vecs = outputs[:, 0, :]  # (num_labels, d_h)
#             else:
#                 value_vecs = torch.mean(outputs, dim=1)  # (num_labels, d_h)
            
#             self.value_lookup[v[0]] = nn.Embedding.from_pretrained(value_vecs, freeze=True)
        
#             nn.init.xavier_uniform_(self.w1[v[0]].weight)
#             nn.init.xavier_uniform_(self.w2[v[0]].weight)
#             nn.init.xavier_uniform_(self.w3[v[0]].weight)
        
        
class ActionPrediction(nn.Module):
    def __init__(self, args):
        super(ActionPrediction, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, trg_hiddens):
        # trg_hiddens: (B, d_h) ([CLS] or mean)
        
        return self.linear(trg_hiddens)  # (B, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        