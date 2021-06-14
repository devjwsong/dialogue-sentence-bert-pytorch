from tqdm import tqdm

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/oos"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    finetune_dir = f"{finetune_dir}/oos"
    if not os.path.isdir(finetune_dir):
        os.makedirs(finetune_dir)

    with open(f"{data_dir}/data_full.json", 'r') as f:
        data = json.load(f)
        
    train_data = data['train'] + data['oos_train']
    valid_data = data['val'] + data['oos_val']
    test_data = data['test'] + data['oos_test']
    
    print("Parsing data contents...")
    intent_class_dict = {}
    train_utters, train_intents, intent_class_dict = parse_infos(train_data, intent_class_dict)
    valid_utters, valid_intents, intent_class_dict = parse_infos(valid_data, intent_class_dict)
    test_utters, test_intents, intent_class_dict = parse_infos(test_data, intent_class_dict)
    
    print("Saving intent class dictionary...")
    with open(f"{finetune_dir}/intent_{args.class_dict_name}.json", 'w') as f:
        json.dump(intent_class_dict, f)
    
    print("Saving data files as pickle...")
    save_file(finetune_dir, 'train', train_utters, train_intents)
    save_file(finetune_dir, 'valid', valid_utters, valid_intents)
    save_file(finetune_dir, 'test', test_utters, test_intents)
    
    print("<Data Anaysis>")
    print("Task: Intent Detection")
    print(f"# of train utterances: {len(train_utters)}")
    print(f"# of valid utterances: {len(valid_utters)}")
    print(f"# of test utterances: {len(test_utters)}")
    print(f"# of classes: {len(intent_class_dict)}")
    
    print("Done.")
    
    
def parse_infos(data, intent_class_dict):
    utters = []
    intents = []
    for pair in tqdm(data):
        utters.append(pair[0])
        intents.append(pair[1])
        
        if pair[1] not in intent_class_dict:
            intent_class_dict[pair[1]] = len(intent_class_dict)
            
    assert len(utters) == len(intents)
        
    return utters, intents, intent_class_dict


def save_file(finetune_dir, prefix, utters, intents):
    with open(f"{finetune_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utters, f)

    with open(f"{finetune_dir}/{prefix}_intents.pickle", 'wb') as f:
        pickle.dump(intents, f)
        