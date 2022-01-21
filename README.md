# dialogue-sentence-bert-pytorch
This repository provides the pre-training & fine-tuning code for the project **"DialogueSentenceBERT: SentenceBERT for More Representative Utterance Embedding via Pre-training on Dialogue Corpus"**.

This modified version of the SentenceBERT[[1\]](#1) is specialized for the dialogue understanding tasks which use an utterance/context embedding vector and an additional task-specific layer, such as intent detection and next system action prediction.

The training objective is also similar to SentBERT, which is based on the Siamese Network[[2\]](#2) structure and optimized via the CosineSimilarity loss function[[3\]](#3).

The model is pre-trained on a large open-domain dialogue corpus called OpenSubtitles and evaluated various task-oriented dialogue tasks, which will be elaborated on below.

<br/>

---

