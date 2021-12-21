from xml.etree.ElementTree import parse
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
from glob import glob
from itertools import chain
from collections import OrderedDict
from natsort import natsort

import argparse
import pickle
import os
import xmltodict
import numpy as np
import random


def parse_file(file):
    f = open(file, 'r')
    data = f.read()
    data = xmltodict.parse(data)
    dial = []

    for s in data["document"]["s"]:
        words = []

        if not isinstance(s, OrderedDict) or "w" not in s:
            continue

        ws = s["w"]
        if not isinstance(ws, list):
            ws = [ws]

        for w in ws:
            text = w["#text"] if "@alternative" not in w else w["@alternative"]
            words.append(text)
        dial.append(" ".join(words))

    dial = dial[args.num_trunc:-1*args.num_trunc]
    
    return dial


def make_without_context(args, utter, tokenizer):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utter))
    seq = [args.cls_id] + token_ids + [args.sep_id]
    if len(seq) > args.max_len:
        seq = seq[:args.max_len]
        seq[-1] = args.sep_id
        
    return seq


def make_with_context(args, contexts, utter, tokenizer, start):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utter))
    for i in range(len(contexts)):
        context = list(chain.from_iterable(contexts[i:]))
        seq_len = len(context) + len(token_ids) + 3
        if seq_len <= args.max_len:
            seq = [args.cls_id] + context + [args.sep_id] + token_ids + [args.sep_id]
            
            return seq, start
        
        start += 1
        
    seq = make_without_context(args, utter, tokenizer)
    
    return seq, start


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--raw_dir', type=str, required=True, help="The directory which contains the raw xml files.")
    parser.add_argument('--data_dir', type=str, default="data/opensubtitles-parsed", help="The parent directory for saving parsed data.")
    parser.add_argument('--bert_ckpt', type=str, default="bert-base-uncased", help="The checkpoint of the BERT to load the tokenizer.")
    parser.add_argument('--lam', type=int, default=2, help="The lambda value for the Poisson distribution.")
    parser.add_argument('--num_trunc', type=int, default=10, help="The number of turns to truncate.")

    args = parser.parse_args()
    
    file_list = glob(f"{args.raw_dir}/xml/en/*/*/*.xml")
    file_list = natsort.natsorted(file_list)
    print(file_list)
    print(f"The total number of files: {len(file_list)}")
    
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
        
    # Load the tokenizer
    config = BertConfig.from_pretrained(args.bert_ckpt)
    tokenizer = BertTokenizer.from_pretrained(args.bert_ckpt)
    vocab = tokenizer.get_vocab()
    args.cls_id = vocab[tokenizer.cls_token]
    args.sep_id = vocab[tokenizer.sep_token]
    args.max_len = config.max_position_embeddings
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("Parsing files...")
    num_parsed_files = 0
    for file in tqdm(file_list):
        dial = parse_file(file)
        if len(dial) == 0:
            continue
            
        preprocessed = []
        idx = len(dial)-1
        while idx >= 0:
            num_contexts = np.random.poisson(args.lam)
            start = max(0, idx-num_contexts)
            context_utters = dial[start:idx]
            contexts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context)) for context in context_utters]
            
            if len(contexts) == 0:
                input_ids = make_without_context(args, dial[idx], tokenizer)
                idx -= 1
            else:
                input_ids, start = make_with_context(args, contexts, dial[idx], tokenizer, start)
                assert start-1 < idx, f"{idx} || {start}"
                idx = start-1
                
            preprocessed.append(input_ids)
            
        with open(f"{args.data_dir}/{num_parsed_files}.pickle", 'wb') as f:
            pickle.dump(preprocessed, f)
        num_parsed_files += 1
        
    print(f"The total number of files: {len(file_list)}.")
    print(f"The parsed number of files: {num_parsed_files}.")
