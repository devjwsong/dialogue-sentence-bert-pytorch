from tqdm import tqdm
from string import digits

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/TM-2-2020"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
    
    # For entity recognition
    entity_dir = f"{finetune_dir}/{args.entity_dir}/taskmaster2"
    if not os.path.isdir(entity_dir):
        os.makedirs(entity_dir)

    entity_class_dict = {'O': 0}
    
    train_utter_dialogues = []
    train_entity_dialogues = []
    valid_utter_dialogues = []
    valid_entity_dialogues = []
    test_utter_dialogues = []
    test_entity_dialogues = []
    
    domain_file_list = [file for file in os.listdir(f"{data_dir}/data") if file.endswith('.json')]
    for domain_file in domain_file_list:
        print(f"Processing {domain_file}...")
        with open(f"{data_dir}/data/{domain_file}", 'r') as f:
            data = json.load(f)
        
        utter_dialogues = []
        entity_dialogues = []
        for dialogue in tqdm(data):
            utter_dialogue = []
            entity_dialogue = []
            
            turns = dialogue['utterances']
            for turn in turns:
                speaker = 'speaker1' if turn['speaker'] == 'USER' else 'speaker2'
                utter = turn['text']
                
                utter_dialogue.append(f"{speaker}:{utter}")
                
                if 'segments' in turn:
                    entity_list, entity_class_dict = find_entities(turn['segments'], entity_class_dict)
                else:
                    entity_list = []
                    
                entity_dialogue.append(entity_list)
                
            utter_dialogues.append(utter_dialogue)
            entity_dialogues.append(entity_dialogue)
                
        train_utters, valid_utters, test_utters = split_data(utter_dialogues, args.train_frac, args.valid_frac)
        train_entities, valid_entities, test_entities = split_data(entity_dialogues, args.train_frac, args.valid_frac)
        
        train_utter_dialogues += train_utters
        train_entity_dialogues += train_entities
        valid_utter_dialogues += valid_utters
        valid_entity_dialogues += valid_entities
        test_utter_dialogues += test_utters
        test_entity_dialogues += test_entities
        
    print("Saving files into pickle formats...")
    save_files('Entity Recognition', entity_dir, args.train_prefix, train_utter_dialogues, train_entity_dialogues, args)
    save_files('Entity Recognition', entity_dir, args.valid_prefix, valid_utter_dialogues, valid_entity_dialogues, args)
    save_files('Entity Recognition', entity_dir, args.test_prefix, test_utter_dialogues, test_entity_dialogues, args)
    
    print("Saving entity class dictionary...")
    with open(f"{entity_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(entity_class_dict, f)
    
    num_train_utters = count_utters(train_utter_dialogues)
    num_valid_utters = count_utters(valid_utter_dialogues)
    num_test_utters = count_utters(test_utter_dialogues)
    
    print("<Data Anaysis>")
    print("Task: Entity Recognition")
    print(f"# of train dialogues: {len(train_utter_dialogues)}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {len(valid_utter_dialogues)}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {len(test_utter_dialogues)}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(entity_class_dict)}")
    
    print("Done.")
    
    
def find_entities(segments, entity_class_dict):
    entity_list = []
    for segment in segments:
        entity_value = segment['text']
        start = segment['start_index']
        end = segment['end_index']
        entity_type = segment['annotations'][0]['name'].translate({ord(k):None for k in digits})
        
        if f"B-{entity_type}" not in entity_class_dict:
            entity_class_dict[f"B-{entity_type}"] = len(entity_class_dict)
            entity_class_dict[f"I-{entity_type}"] = len(entity_class_dict)
            
        entity_list.append((entity_type, entity_value, start, end))
        
    return entity_list, entity_class_dict
    

def split_data(dialogues, train_frac, valid_frac):    
    train_dialogues = dialogues[:int(len(dialogues) * train_frac)]
    remained_dialogues = dialogues[int(len(dialogues) * train_frac):]
    
    f = valid_frac / (1.0 - train_frac)
    valid_dialogues = remained_dialogues[:int(len(remained_dialogues) * f)]
    test_dialogues = remained_dialogues[int(len(remained_dialogues) * f):]
    
    return train_dialogues, valid_dialogues, test_dialogues


def save_files(task_name, save_dir, prefix, utter_dialogues, label_dialogues, args):
    print(f"Saving {prefix} data for {task_name} as pickle...")
    with open(f"{save_dir}/{prefix}_{args.utter_name}.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{save_dir}/{prefix}_{args.label_name}.pickle", 'wb') as f:
        pickle.dump(label_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        