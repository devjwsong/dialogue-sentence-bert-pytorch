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

<br/>

---

<a id="1">[1]</a> Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084. <a href="https://arxiv.org/pdf/1908.10084.pdf">https://arxiv.org/pdf/1908.10084.pdf</a>

<a id="2">[2]</a> Koch, G., Zemel, R., & Salakhutdinov, R. (2015, July). Siamese neural networks for one-shot image recognition. In ICML deep learning workshop (Vol. 2). <a href="https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf">https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf</a>

<a id="3">[3]</a> COSINEEMBEDDINGLOSS. <a href="https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html">https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html</a>

<a id="4">[4]</a> Lison, P., & Tiedemann, J. (2016). Opensubtitles2016: Extracting large parallel corpora from movie and tv subtitles. <a href="http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf">http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf</a>

<a id="5">[5]</a> SongStudio. <a href="https://songstudio.info/">https://songstudio.info</a>