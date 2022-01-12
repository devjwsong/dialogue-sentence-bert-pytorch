python src/data_utils/finetune_data_process.py \
    --raw_dir="data/raw" \
    --finetune_dir="data/finetune" \
    --train_frac=0.8 \
    --valid_frac=0.1 \
    --train_prefix="train" \
    --valid_prefix="valid" \
    --test_prefix="test" \
    --class_dict_name="class_dict"
