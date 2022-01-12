from tqdm import tqdm

import os
import json
import pickle
import csv


def process_data(args):
    data_dir = f"{args.data_dir}/{args.raw_dir}/banking77"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    save_dir = f"{args.finetune_dir}/banking77"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    with open(f"{data_dir}/categories.json", 'r') as f:
        categories = json.load(f)
    
    print("Making intent class dict...")
    intent_class_dict = {}
    for i, intent in enumerate(tqdm(categories)):
        intent_class_dict[intent] = i
        
    print("Parsing data contents...")
    train_utters, train_intents = parse_infos(data_dir, "train", intent_class_dict)
    test_utters, test_intents = parse_infos(data_dir, "test", intent_class_dict)
    
    print("Splitting train/valid set...")
    train_utters, train_intents, valid_utters, valid_intents = split_data(train_utters, train_intents)
                
    print("Saving intent class dictionary...")
    with open(f"{save_dir}/intent_{args.class_dict_name}.json", 'w') as f:
        json.dump(intent_class_dict, f)
    
    print("Saving data files as pickle...")
    save_file(save_dir, 'train', train_utters, train_intents)
    save_file(save_dir, 'valid', valid_utters, valid_intents)
    save_file(save_dir, 'test', test_utters, test_intents)
    
    print("<Data Anaysis>")
    print("Task: Intent Detection")
    print(f"# of train utterances: {len(train_utters)}")
    print(f"# of valid utterances: {len(valid_utters)}")
    print(f"# of test utterances: {len(test_utters)}")
    print(f"# of classes: {len(intent_class_dict)}")
    
    print("Done.")
    
    
def parse_infos(data_dir, data_type, intent_class_dict):
    utters = []
    intents = []
    
    with open(f"{data_dir}/{data_type}.csv", 'r') as f:
        lines = list(csv.reader(f))
            
    for l, line in enumerate(tqdm(lines)):
        if l > 0:
            intent = line[1]
            utter = line[0]
            
            utters.append(utter)
            intents.append(intent)
            
    return utters, intents


def split_data(utters, intents, frac=0.9):
    i2u = {}
    for i, intent in enumerate(intents):
        if intent not in i2u:
            i2u[intent] = []
            
        i2u[intent].append(utters[i])
        
    first_utters, first_intents = [], []
    second_utters, second_intents = [], []
    
    for intent, utter_list in i2u.items():
        first_len = int(len(utter_list) * frac)
        second_len = len(utter_list) - first_len
        first_utters += utter_list[:first_len]
        second_utters += utter_list[first_len:]
        first_intents += [intent] * first_len
        second_intents += [intent] * second_len
        
    assert len(first_utters) == len(first_intents)
    assert len(second_utters) == len(second_intents)
    
    return first_utters, first_intents, second_utters, second_intents


def save_file(save_dir, prefix, utters, intents):
    with open(f"{save_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utters, f)

    with open(f"{save_dir}/{prefix}_intents.pickle", 'wb') as f:
        pickle.dump(intents, f)
        