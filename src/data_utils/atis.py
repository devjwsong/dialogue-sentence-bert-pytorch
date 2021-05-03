from tqdm import tqdm

import os
import json
import pickle


def process_data(args, processed_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/atis"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    processed_dir = f"{processed_dir}/atis"
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)
        
    print("Parsing data contents...")
    train_utters, train_intents, intent_class_dict = parse_infos(data_dir, "atis.train.pkl")
    test_utters, test_intents, _ = parse_infos(data_dir, "atis.test.pkl")
    
    print("Splitting train/valid set...")
    train_utters, train_intents, valid_utters, valid_intents = split_data(train_utters, train_intents)
    
    print("Saving intent class dictionary...")
    with open(f"{processed_dir}/intent_{args.class_dict_name}.json", 'w') as f:
        json.dump(intent_class_dict, f)
    
    print("Saving data files as pickle...")
    save_file(processed_dir, 'train', train_utters, train_intents)
    save_file(processed_dir, 'valid', valid_utters, valid_intents)
    save_file(processed_dir, 'test', test_utters, test_intents)
    
    print("<Data Anaysis>")
    print("Task: Intent Detection")
    print(f"# of train utterances: {len(train_utters)}")
    print(f"# of valid utterances: {len(valid_utters)}")
    print(f"# of test utterances: {len(test_utters)}")
    print(f"# of classes: {len(intent_class_dict)}")
    
    print("Done.")
    
    
def parse_infos(data_dir, file):
    with open(f"{data_dir}/{file}", 'rb') as f:
        data = pickle.load(f)
        
    vocab = data[1]['token_ids']
    intent_class_dict = data[1]['intent_ids']
    
    i2w = {}
    for token, idx in vocab.items():
        i2w[idx] = token
    i2i = {}
    for intent, idx in intent_class_dict.items():
        i2i[idx] = intent
        
    queries = data[0]['query']
    labels = data[0]['intent_labels']
    
    utters = []
    intents = []
    for q, query in enumerate(tqdm(queries)):
        utter = ' '.join([i2w[token_id] for token_id in query][1:-1])
        intent = i2i[labels[q][0]]
        
        utters.append(utter)
        intents.append(intent)
    
    assert len(utters) == len(intents)
    
    return utters, intents, intent_class_dict


def save_file(processed_dir, prefix, utters, intents):
    with open(f"{processed_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utters, f)

    with open(f"{processed_dir}/{prefix}_intents.pickle", 'wb') as f:
        pickle.dump(intents, f)
        

def split_data(utters, intents, frac=0.9):
    i2u = {}
    for i, intent in enumerate(intents):
        if intent not in i2u:
            i2u[intent] = []
            
        i2u[intent].append(utters[i])
        
    first_utters, first_intents = [], []
    second_utters, second_intents = [], []
    
    for intent, utter_list in i2u.items():
        first_utters += utter_list[:int(len(utter_list) * frac)]
        second_utters += utter_list[int(len(utter_list) * frac):]
        first_intents += [intent] * len(first_utters)
        second_intents += [intent] * len(second_utters)
        
    assert len(first_utters) == len(first_intents)
    assert len(second_utters) == len(second_intents)
    
    return first_utters, first_intents, second_utters, second_intents
    