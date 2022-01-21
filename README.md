# dialogue-sentence-bert-pytorch
This repository provides the pre-training & fine-tuning code for the project **"DialogueSentenceBERT: SentenceBERT for More Representative Utterance Embedding via Pre-training on Dialogue Corpus"**.

This modified version of the SentenceBERT[[1]](#1) is specialized for the dialogue understanding tasks which use an utterance/context embedding vector and an additional task-specific layer, such as intent detection and next system action prediction.

The training objective is also similar to SentBERT, which is based on the Siamese Network[[2]](#2) structure and optimized via the CosineSimilarity loss function[[3]](#3).

The model is pre-trained on a large open-domain dialogue corpus called OpenSubtitles[[4]](#4) and evaluated various task-oriented dialogue tasks, which will be elaborated on below.

<br/>

---

### Implementation details

This document only specify how to run the scripts in order to pre-train and evaluate the DialogueSentenceBERT.

The details of the overall ideas and implementation are explained in my blog, SongStudio[[5]](#5).

You are always welcomed to visit the posts regarding this repository as follows.

- https://songstudio.info/tech/tech-38
- https://songstudio.info/tech/tech-39

<br/>

---

### Arguments

**Arguments for parsing the OpenSubtitles raw files**

| Argument      | Type  | Description                                       | Default                       |
| ------------- | ----- | ------------------------------------------------- | ----------------------------- |
| `--seed`      | `int` | The random seed.                                  | `0`                           |
| `--raw_dir`   | `str` | The directory which contains the raw xml files.   | *YOU SHOULD SPECIFY.*         |
| `--data_dir`  | `str` | The parent directory for saving parsed data.      | `"data/opensubtitles-parsed"` |
| `--bert_ckpt` | `str` | The checkpoint of the BERT to load the tokenizer. | `"bert-base-uncased"`         |
| `--lam`       | `int` | The lambda value for the Poisson distribution.    | `2`                           |
| `--num_trunc` | `int` | The number of turns to truncate.                  | `20`                          |

<br/>

**Arguments for pre-processing of the parsed OpenSubtitles files**

| Argument       | Type  | Description                                        | Default                       |
| -------------- | ----- | -------------------------------------------------- | ----------------------------- |
| `--seed`       | `int` | The random seed.                                   | `0`                           |
| `--parsed_dir` | `str` | The parent directory for saving parsed data.       | `"data/opensubtitles-parsed"` |
| `--keep_ratio` | `str` | The ratio of sampled to be kept as the same pairs. | `0.66`                        |
| `--save_dir`   | `str` | The directory for saving pre-train data files.     | `"data/pretrain"`             |
| `--group_size` | `int` | The maximum number of samples in each file.        | `100000`                      |

<br/>

**Arguments for data pre-shuffling before pre-training**

| Argument         | Type  | Description                                               | Default                    |
| ---------------- | ----- | --------------------------------------------------------- | -------------------------- |
| `--seed`         | `int` | The random seed.                                          | `0`                        |
| `--pretrain_dir` | `str` | The directory which contains the pre-train data files.    | `"data/pretrain"`          |
| `--shuffled_dir` | `str` | The directory which will contain the shuffled data files. | `"data/pretrain-shuffled"` |
| `--num_files`    | `int` | The directory for saving pre-train data files.            | `128`                      |

<br/>

**Arguements for pre-training**

| Argument             | Type    | Description                                                  | Default               |
| -------------------- | ------- | ------------------------------------------------------------ | --------------------- |
| `--default_root_dir` | `str`   | The default directory for logs & checkpoints.                | `"./"`                |
| `--shuffled_dir`     | `str`   | The directory which contains the shuffled pre-train data files." | `"pretrain-shuffled"` |
| `--num_epochs`       | `int`   | The number of total epochs.                                  | `1`                   |
| `--batch_size`       | `int`   | The batch size assigned to each GPU.                         | `32`                  |
| `--num_workers`      | `int`   | The number of workers for data loading.                      | `4`                   |
| `--learning_rate`    | `float` | The initial learning rate.                                   | `2e-5`                |
| `--warmup_ratio`     | `float` | The warmup step ratio.                                       | `0.1`                 |
| `--save_interval`    | `int`   | The training step interval to save checkpoints.              | `50000`               |
| `--log_interval`     | `int`   | The training step interval to write logs.                    | `10000`               |
| `--max_grad_norm`    | `float` | The max gradient for gradient clipping.                      | `1.0`                 |
| `--seed`             | `int`   | The random seed number.                                      | `0`                   |
| `--pooling`          | `str`   | Pooling method: CLS/Mean/Max.                                | *YOU SHOULD SPECIFY.* |
| `--ckpt_dir`         | `str`   | If only training from a specific checkpoint... (also convbert) | *YOU MIGHT SPECIFY.*  |
| `--gpus`             | `str`   | The indices of GPUs to use.                                  | `"0, 1, 2, 3"`        |
| `--amp_level`        | `str`   | The optimization level to use for 16-bit GPU precision.      | `O1`                  |
| `--num_nodes`        | `int`   | The number of machine.                                       | `1`                   |

<br/>

**Arguments for extracting pre-trained checkpoint**

| Argument             | Type  | Description                                   | Default               |
| -------------------- | ----- | --------------------------------------------- | --------------------- |
| `--default_root_dir` | `str` | The default directory for logs & checkpoints. | `"./"`                |
| `--log_idx`          | `int` | The lightning log index.                      | *YOU SHOULD SPECIFY.* |
| `--ckpt_file`        | `str` | The checkpoint file to extract.               | *YOU SHOULD SPECIFY.* |

<br/>

**Arguments for pre-processing of the raw fine-tuning data**

| Argument            | Type    | Description                                                  | Default           |
| ------------------- | ------- | ------------------------------------------------------------ | ----------------- |
| `--raw_dir`         | `str`   | The directory path for raw data files.                       | `"data/raw"`      |
| `--finetune_dir`    | `str`   | The directory path to processed finetune data files.         | `"data/finetune"` |
| `--train_frac`      | `float` | The ratio of the conversations to be included in the train set. | `0.8`             |
| `--valid_frac`      | `float` | The ratio of the conversations to be included in the valid set. | `0.1`             |
| `--train_prefix`    | `str`   | The prefix of file name related to train set.                | `"train"`         |
| `--valid_prefix`    | `str`   | The prefix of file name related to validation set.           | `"valid"`         |
| `--test_prefix`     | `str`   | The prefix of file name related to test set.                 | `"test"`          |
| `--class_dict_name` | `str`   | The name of class dictionary json file.                      | `"class_dict"`    |

<br/>

**Arguments for fine-tuning**

| Argument              | Type           | Description                                                  | Default               |
| --------------------- | -------------- | ------------------------------------------------------------ | --------------------- |
| `--task`              | `str`          | The name of the task.                                        | *YOU SHOULD SPECIFY.* |
| `--dataset`           | `str`          | The name of the dataset.                                     | *YOU SHOULD SPECIFY.* |
| `--cached_dir`        | `str`          | The directory for pre-processed data pickle files after fine-tuning. | `"cached"`            |
| `--finetune_dir`      | `str`          | The directory of finetuning data files.                      | `"data/finetune"`     |
| `--class_dict_prefix` | `str`          | The prefix of class dictionary json file.                    | `"class_dict"`        |
| `--train_prefix`      | `str`          | The prefix of file name related to train set.                | `"train"`             |
| `--valid_prefix`      | `str`          | The prefix of file name related to validation set.           | `"valid"`             |
| `--test_prefix`       | `str`          | The prefix of file name related to test set.                 | `"test"`              |
| `--max_turns`         | `int`          | The maximum number of dialogue turns.                        | `1`                   |
| `--num_epochs`        | `int`          | The number of total epochs.                                  | `20`                  |
| `--batch_size`        | `int`          | The batch size in one process.                               | `16`                  |
| `--num_workers`       | `int`          | The number of workers for data loading.                      | `4`                   |
| `--max_encoder_len`   | `int`          | The maximum length of a sequence.                            | `512`                 |
| `--learning_rate`     | `float`        | The initial learning rate.                                   | `5e-5`                |
| `--warmup_prop`       | `float`        | The warmup step proportion.                                  | `0.0`                 |
| `--max_grad_norm`     | `float`        | The max gradient for gradient clipping.                      | `1.0`                 |
| `--sigmoid_threshold` | `float`        | The sigmoid threshold for action prediction task.            | `0.5`                 |
| `--cached`            | `"store_true"` | Using the cached data or not? (if exist...)                  | *YOU MIGHT SPECIFY.*  |
| `--seed`              | `int`          | The random seed.                                             | `0`                   |
| `--model_name`        | `str`          | The encoder model to test.                                   | *YOU SHOULD SPECIFY.* |
| `--pooling`           | `str`          | Pooling method: CLS/Mean/Max.                                | *YOU SHOULD SPECIFY.* |
| `--gpu`               | `str`          | The indext of GPU to use.                                    | `"0"`                 |
| `--ckpt_dir`          | `str`          | If only training from a specific checkpoint... (also convbert & dialogsentbert) | *YOU MIGHT SPECIFY.*  |

<br/>

---

<a id="1">[1]</a> Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084. <a href="https://arxiv.org/pdf/1908.10084.pdf">https://arxiv.org/pdf/1908.10084.pdf</a>

<a id="2">[2]</a> Koch, G., Zemel, R., & Salakhutdinov, R. (2015, July). Siamese neural networks for one-shot image recognition. In ICML deep learning workshop (Vol. 2). <a href="https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf">https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf</a>

<a id="3">[3]</a> COSINEEMBEDDINGLOSS. <a href="https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html">https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html</a>

<a id="4">[4]</a> Lison, P., & Tiedemann, J. (2016). Opensubtitles2016: Extracting large parallel corpora from movie and tv subtitles. <a href="http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf">http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf</a>

<a id="5">[5]</a> SongStudio. <a href="https://songstudio.info/">https://songstudio.info</a>