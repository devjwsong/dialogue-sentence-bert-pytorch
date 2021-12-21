python src/data_utils/pretrain/parse_opensubtitles.py \
    --seed=0 \
    --raw_dir="data/raw/OpenSubtitles" \
    --data_dir="data/opensubtitles-parsed" \
    --bert_ckpt="bert-base-uncased" \
    --lam=2 \
    --num_trunc=20
