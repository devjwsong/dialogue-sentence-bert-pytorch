from tqdm import tqdm

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/dstc2"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
        
    finetune_dir = f"{finetune_dir}/dstc2"
    if not os.path.isdir(finetune_dir):
        os.makedirs(finetune_dir)
    
    action_class_dict = {}
    
    with open(f"{data_dir}/traindev/scripts/config/dstc2_train.flist") as f:
        train_list = f.readlines()
    with open(f"{data_dir}/traindev/scripts/config/dstc2_dev.flist") as f:
        valid_list = f.readlines()
    with open(f"{data_dir}/test/scripts/config/dstc2_test.flist") as f:
        test_list = f.readlines()
        
    train_utter_dialogues, train_action_dialogues, action_class_dict = load_dialogues(f"{data_dir}/traindev/data", train_list, action_class_dict)
    valid_utter_dialogues, valid_action_dialogues, action_class_dict = load_dialogues(f"{data_dir}/traindev/data", valid_list, action_class_dict)
    test_utter_dialogues, test_action_dialogues, action_class_dict = load_dialogues(f"{data_dir}/test/data", test_list, action_class_dict)
    
    save_file(finetune_dir, args.train_prefix, train_utter_dialogues, train_action_dialogues)
    save_file(finetune_dir, args.valid_prefix, valid_utter_dialogues, valid_action_dialogues)
    save_file(finetune_dir, args.test_prefix, test_utter_dialogues, test_action_dialogues)
    
    print("Saving action class dictionary...")
    with open(f"{finetune_dir}/action_{args.class_dict_name}.json", 'w') as f:
        json.dump(action_class_dict, f)
    
    num_train_utters = count_utters(train_utter_dialogues)
    num_valid_utters = count_utters(valid_utter_dialogues)
    num_test_utters = count_utters(test_utter_dialogues)
    
    print("<Data Anaysis>")
    print("Task: Action Prediction")
    print(f"# of train dialogues: {len(train_utter_dialogues)}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {len(valid_utter_dialogues)}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {len(test_utter_dialogues)}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(action_class_dict)}")
    
    print("Done.")


def load_dialogues(data_dir, data_list, action_class_dict):
    utter_dialogues = []
    action_dialogues = []
    
    for data_name in tqdm(data_list):
        dir1 = data_name.strip().split('/')[0]
        dir2 = data_name.strip().split('/')[1]
        
        assert os.path.isdir(f"{data_dir}/{dir1}/{dir2}")
        
        utter_dialogue = []
        action_dialogue = []
        
        with open(f"{data_dir}/{dir1}/{dir2}/log.json") as f:
            log = json.load(f)
        with open(f"{data_dir}/{dir1}/{dir2}/label.json") as f:
            label = json.load(f)
            
        log_turns = log['turns']
        label_turns = label['turns']

        for t, turn in enumerate(log_turns):
            sys_utter = turn['output']['transcript']
            usr_utter = label_turns[t]['transcription']

            utter_dialogue.append(f"sys:{sys_utter}")
            utter_dialogue.append(f"usr:{usr_utter}")

            sys_acts = turn['output']['dialog-acts']  # System side
            usr_acts = label_turns[t]['semantics']['json']  # User side

            sys_action_list, action_class_dict = find_actions(sys_acts, action_class_dict, speaker='sys')
            usr_action_list, action_class_dict = find_actions(usr_acts, action_class_dict, speaker='usr')

            action_dialogue += [sys_action_list, usr_action_list]

        assert len(utter_dialogue) == len(action_dialogue)

        utter_dialogues.append(utter_dialogue)
        action_dialogues.append(action_dialogue)
    
    assert len(utter_dialogues) == len(action_dialogues)
    
    return utter_dialogues, action_dialogues, action_class_dict


def find_actions(dialog_acts, action_class_dict, speaker):
    action_list = []

    for obj in dialog_acts:
        action = obj['act']

        if action not in action_class_dict and speaker == 'sys':
            action_class_dict[action] = len(action_class_dict)

        action_list.append(("", action))
        
    return list(set(action_list)), action_class_dict


def save_file(finetune_dir, prefix, utter_dialogues, action_dialogues):
    with open(f"{finetune_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{finetune_dir}/{prefix}_actions.pickle", 'wb') as f:
        pickle.dump(action_dialogues, f)
        
        
def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        
