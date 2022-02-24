# mipt-nlp2022
NLP course, MIPT

### Course instructors
Anton Emelianov (login-const@mail.ru, @king_menin), Albina Akhmetgareeva (albina.akhmetgareeva@gmail.com)

Videos [here](https://drive.google.com/drive/folders/1CDQsHx53en5punmtB5I94A4NI2pJY9Wm?usp=sharing)

## Mark
```math
final_mark=sum_i (max_score(HW_i)) / count(HWs) * 10, i==1..3
```

## Lecture schedule

#### Week 1

* Lecture: [Intro to NLP](lectures/L1.Intro2NLP.pdf)
* Practical: [Text preprocessing](seminars/sem1/sem1_basic_text_processing.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/king-menin/mipt-nlp2022/blob/master/seminars/sem1/sem1_basic_text_processing.ipynb)
* [Video](https://drive.google.com/file/d/1hx3EHLtIsOspEjMDZz9I0ENH3P9DIsrf/view?usp=sharing)

#### Week 2

* Lecture: Word embeddings

Distributional semantics. Count-based (pre-neural) methods. Word2Vec: learn vectors. GloVe: count, then learn. N-gram (collocations)
RusVectores. t-SNE.
* Practical: word2vec, fasttext
* [HW1](HWs/hw1.ipynb)
* Video: [lecture](https://drive.google.com/file/d/1LQEVudRMccfiIj5igVeNg2qqg9EwRJ7b/view?usp=sharing), [seminar](https://drive.google.com/file/d/1yoXQXRmEvhBUl0iPJgVXFYv9KwyLAzUA/view?usp=sharing)

#### Week 3

* Lecture: RNN + CNN, Text classification

Neural Language Models: Recurrent Models, Convolutional Models. Text classification (architectures)
* Practical: [Classification with LSTM, CNN](seminars/sem3/sem3_classification.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/king-menin/mipt-nlp2021/blob/master/seminars/sem3/sem3_classification.ipynb)
* Video

#### Week 4

* Lecture: Language modelling and NER

Task description, methods (Markov Model, RNNs), evaluation (perplexity), Sequence Labelling (NER, pos-tagging, chunking etc.) N-gram language models, HMM, MEMM, CRF
* Practical: NER
* HW2
* Video

#### Week 5

* Lecture: Machine translation, Seq2seq, Attention, Transformers

Basics: Encoder-Decoder framework, Inference (e.g., beam search),  Eval (bleu).
Attention: general, score functions, models. Bahdanau and Luong models. 
Transformer: self-attention, masked self-attention, multi-head attention.

* Video

#### Week 6

* Lecture: Transfer learning in NLP

Bertology (BERT, GPT-s, t5, etc.), Subword Segmentation (BPE), Evaluation of big LMs.
* Practical: transformers models for classification task
* Practical: Transfer learning
* Video: Part1

#### Week 7

Lecture & Practical: How to train big models? Part1. Distributed training, Part2. RuGPT3 Training

Training Multi-Billion Parameter Language Models. Model Parallelism. Data Parallelism.

* Practical: DDP example
* Video
* HW3

#### Week 8

* Lecture: Syntax parsing
* Practical: Syntax
* Video

#### Week 9

* Lecture: Question answering
* Practical: [seminar QA, seminar chat-bots
* Video

Squads (one-hop, multi-hop), architectures, retrieval and search, chat-bots


#### Week 10

* Lecture: Summarization, simplification, paraphrasing
* Practical: summarization seminar
* Video

#### Week 11

* Lecture: Knowledge Distillation in NLP
* Video



## Recommended Resources
### En

* [ruder.io](https://ruder.io/)
* [Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/)
* [Курс Лауры Каллмайер по МО для NLP](https://user.phil.hhu.de/~kallmeyer/MachineLearning/index.html)
* [Курс Нильса Раймерса по DL для NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial)
* [Курс в Оксфорде по DL для NLP](https://github.com/UKPLab/deeplearning4nlp-tutorial)
* [Курс в Стенфорде по DL для NLP](http://cs224d.stanford.edu)
* [Reinforcment Learning for NLP](https://github.com/jiyfeng/rl4nlp)


### На русском (и про русский, в основном)

* [Курс nlp в яндексе](https://github.com/yandexdataschool/nlp_course)
* [НКРЯ](http://ruscorpora.ru)
* [Открытый корпус](http://opencorpora.org)
* [Дистрибутивные семантические модели для русского языка](http://rusvectores.org/ru/)
* [Морфология](https://tech.yandex.ru/mystem/)
* [Синтаксис](https://habrahabr.ru/post/317564/)
* [Томита-парсер](https://tech.yandex.ru/tomita/)
* [mathlingvo](http://mathlingvo.ru)
* [nlpub](https://nlpub.ru)
* [Text Visualisation browser](http://textvis.lnu.se)


## Literature

* Manning, Christopher D., and Hinrich Schütze. Foundations of statistical natural language processing. Vol. 999. Cambridge: MIT press, 1999.
* Martin, James H., and Daniel Jurafsky. "Speech and language processing." International Edition 710 (2000): 25.
* Cohen, Shay. "Bayesian analysis in natural language processing." Synthesis Lectures on Human Language Technologies 9, no. 2 (2016): 1-274.
* Goldberg, Yoav. "Neural Network Methods for Natural Language Processing." Synthesis Lectures on Human Language Technologies 10, no. 1 (2017): 1-309.

