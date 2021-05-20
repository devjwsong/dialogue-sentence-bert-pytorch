for max_turn in 1 3 5
do
    python src/main.py \
        --task="action" \
        --dataset="multiwoz" \
        --cache_dir="cached" \
        --data_dir="data" \
        --processed_dir="processed" \
        --class_dict_name="class_dict" \
        --train_prefix="train" \
        --valid_prefix="valid" \
        --test_prefix="test" \
        --max_turns=${max_turn} \
        --num_epochs=10 \
        --batch_size=16 \
        --num_workers=4 \
        --max_encoder_len=512 \
        --learning_rate=5e-5 \
        --warmup_prop=0.1 \
        --max_grad_norm=1.0 \
        --sigmoid_threshold=0.5 \
        --seed=0 \
        --model_name=MODEL_NAME \
        --gpu=GPU \
        --num_nodes=1

    python src/main.py \
        --task="action" \
        --dataset="dstc2" \
        --cache_dir="cached" \
        --data_dir="data" \
        --processed_dir="processed" \
        --class_dict_name="class_dict" \
        --train_prefix="train" \
        --valid_prefix="valid" \
        --test_prefix="test" \
        --max_turns=${max_turn} \
        --num_epochs=10 \
        --batch_size=16 \
        --num_workers=4 \
        --max_encoder_len=512 \
        --learning_rate=5e-5 \
        --warmup_prop=0 \
        --max_grad_norm=1.0 \
        --sigmoid_threshold=0.5 \
        --seed=0 \
        --model_name=MODEL_NAME \
        --gpu=GPU \
        --num_nodes=1
        
    python src/main.py \
        --task="action" \
        --dataset="sim" \
        --cache_dir="cached" \
        --data_dir="data" \
        --processed_dir="processed" \
        --class_dict_name="class_dict" \
        --train_prefix="train" \
        --valid_prefix="valid" \
        --test_prefix="test" \
        --max_turns=${max_turn} \
        --num_epochs=10 \
        --batch_size=16 \
        --num_workers=4 \
        --max_encoder_len=512 \
        --learning_rate=2e-4 \
        --warmup_prop=0 \
        --max_grad_norm=1.0 \
        --sigmoid_threshold=0.5 \
        --seed=0 \
        --model_name=MODEL_NAME \
        --gpu=GPU \
        --num_nodes=1
done