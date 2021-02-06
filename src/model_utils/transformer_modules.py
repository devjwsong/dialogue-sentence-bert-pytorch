from transformers import *


# bert_name = "bert-base-uncased"
# todbert_name = "TODBERT/TOD-BERT-JNT-V1"


def load_encoder(args):
    if 'tod' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("TODBERT/" + args.model_name)
        encoder = AutoModel.from_pretrained("TODBERT/" + args.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
        encoder = BertModel.from_pretrained(args.model_name)
        
    encoder_config = encoder.config
    args.max_len = min(args.max_len, encoder_config.max_position_embeddings)
    args.hidden_size = encoder_config.hidden_size
    args.bos_token = '[CLS]'
    args.eos_token = '[SEP]'
    args.pad_token = '[PAD]'
    args.speaker1_token = '[usr]'
    args.speaker2_token = '[sys]'
    
    special_tokens = {
        'additional_special_tokens': [args.speaker1_token, args.speaker2_token]
    }
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    encoder.resize_token_embeddings(args.vocab_size)
    
    return tokenizer, encoder, args


def load_decoder(args):
    if 'dialo' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/" + args.model_name)
        decoder = AutoModelWithLMHead.from_pretrained("microsoft/" + args.model_name).transformer
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        decoder = GPT2Model.from_pretrained(args.model_name)
        
    decoder_config = decoder.config
    args.max_len = min(args.max_len, decoder_config.max_position_embeddings)
    args.hidden_size = decoder_config.n_embd
    args.bos_token = '[BOS]'
    args.eos_token = '[EOS]'
    args.pad_token = '[PAD]'
    args.speaker1_token = '[usr]'
    args.speaker2_token = '[sys]'
    
    special_tokens = {
        'bos_token': args.bos_token,
        'eos_token': args.eos_token,
        'pad_token': args.pad_token,
        'additional_special_tokens': [args.speaker1_token, args.speaker2_token]
    }
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    decoder.resize_token_embeddings(args.vocab_size)
    
    return tokenizer, decoder, args
    