from tqdm import tqdm

import os
import json
import pickle


def process_data(args, processed_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/TM-3-2020/data"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
    
    processed_dir = f"{processed_dir}/taskmaster3"
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    entity_class_dict = {'O': 0}
    
    utter_dialogues = []
    entity_dialogues = []
    
    file_list = [file for file in os.listdir(f"{data_dir}") if file.endswith('.json')]
    for file in file_list:
        print(f"Processing {file}...")
        with open(f"{data_dir}/{file}") as f:
            data = json.load(f)
            
        for obj in tqdm(data):
            turns = obj['utterances']
            utter_dialogue = []
            entity_dialogue = []
            for turn in turns:
                speaker = 'speaker1' if turn['speaker'] == 'user' else 'speaker2'
                utter = turn['text']
                utter_dialogue.append(f"{speaker}:{utter}")
                
                if 'segments' in turn:
                    entity_list, entity_class_dict = find_entities(turn['segments'], entity_class_dict)
                else:
                    entity_list = []
                entity_dialogue.append(entity_list)
        
            utter_dialogues.append(utter_dialogue)
            entity_dialogues.append(entity_dialogue)
            
    print("Splitting data...")
    train_utter_dialogues, valid_utter_dialogues, test_utter_dialogues = \
        split_data(utter_dialogues, args.train_frac, args.valid_frac)
    train_entity_dialogues, valid_entity_dialogues, test_entity_dialogues = \
        split_data(entity_dialogues, args.train_frac, args.valid_frac)
        
    print("Saving files into pickle formats...")
    save_file(processed_dir, args.train_prefix, train_utter_dialogues, train_entity_dialogues)
    save_file(processed_dir, args.valid_prefix, valid_utter_dialogues, valid_entity_dialogues)
    save_file(processed_dir, args.test_prefix, test_utter_dialogues, test_entity_dialogues)
    
    print("Saving entity class dictionary...")
    with open(f"{processed_dir}/entity_{args.class_dict_name}.json", 'w') as f:
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
        start = segment['start_index']
        end = segment['end_index']
        entity_value = segment['text']
        entity_type = segment['annotations'][0]['name']
        
        if f"B-{entity_type}" not in entity_class_dict:
            entity_class_dict[f"B-{entity_type}"] = len(entity_class_dict)
            entity_class_dict[f"I-{entity_type}"] = len(entity_class_dict)
            
        entity_tuple = (entity_type, entity_value, start, end)
        entity_list.append(entity_tuple)
                
    return entity_list, entity_class_dict


def split_data(dialogues, train_frac, valid_frac):
    train_dialogues = dialogues[:int(len(dialogues) * train_frac)]
    remained_dialogues = dialogues[int(len(dialogues) * train_frac):]
    
    f = valid_frac / (1.0-train_frac)
    valid_dialogues = remained_dialogues[:int(len(remained_dialogues) * f)]
    test_dialogues = remained_dialogues[int(len(remained_dialogues) * f):]
    
    return train_dialogues, valid_dialogues, test_dialogues


def save_file(processed_dir, prefix, utter_dialogues, entity_dialogues):
    with open(f"{processed_dir}/{prefix}_utter.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)

    with open(f"{processed_dir}/{prefix}_entity.pickle", 'wb') as f:
        pickle.dump(entity_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        