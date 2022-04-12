# Question Answering Task

Note that we do not have access to the hyperparameters chosen by the authors of data2vec to finetune their model on 
GLUE benchmark. Moreover, we have chosen to look at F1-score and EM score instead of accuracy as we feel that it makes 
more sense.

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It has been developed by Chloé 
Sekkat and can be viewed as a Python package whose main functions/classes can be found in the ``__init__.py``.

## Task description

In this section, we are interested by the capacities of the embeddings produced by data2vec. We evaluate its capacities 
on extractive Question Answering (select minimal span answer within a context) on SQuADv2 dataset. The latter is
a unilingual (English) dataset. The two main metrics used are the F1 score and the Exact Match (EM) score. The obtained
F1-scores are being compared to BERT-like models (BERT, DistilBERT, XLM-RoBERTa and mBERT) and CANINE. Note that mBERT, 
XLM-RoBERTa and CANINE were pre-trained on multilingual data. \textbf{Therefore data2vec is only directly comparable to 
BERT and RoBERTa, but it is still interesting to get the performances on other models on the same task.}

A second step of our analysis is to assess data2vec abilities to handle noisy inputs, especially noisy questions. The 
robustness to noise of such system is imperative. It is highly probable that in real life settings, the Automatic Speech 
Recognition system (ASR) or the human typing the question actually do not produce qualitative text in the sense that the 
written-translation might be flawed (typos, misspellings, grammatical errors, etc). 

The last step is not trutly focused on data2vec but rather on multi-lingual models and their capacities of generalization 
in the context of zero-shot transfer. We decided to include this experiment on zero-shot transfer on a multi-lingual 
dataset here even if it is biased because it allows to compare models pre-trained on multilingual data and that we are 
interested by the zero-shot transfer task. Models are finetuned on an English dataset and then directly evaluated on a 
multi-lingual dataset with 11 languages of various morphologies (XQuAD). Intuitively, only the models trained on 
multilingual data will be able to generalize. Therefore this experience will only allow us to compare mBERT, XLM-RoBERTa 
and CANINE. For the sake of completeness we will nonetheless add the scores obtained by data2vec, BERT, DistilBERT and RoBERTa. 

## Structure

```
question_answering  
├── __init__.py
├── main.py                                         # Main script to finetuned models on SQuADv2/SQuADv1.1 dataset
├── requirements.txt
├── question_answering                              # Main Python package
    ├── dataset_tokenizers
        ├── __init__.py
        ├── dataset_character_based_tokenizer.py    # DatasetTokenizer for character-based models (e.g. CANINE)
        └── dataset_token_based_tokenizer.py        # DatasetTokenizer for token-based models (e.g. BERT, data2vec)
    ├── processing  
        ├── __init__.py 
        ├── preprocessor.py                         # Basic dataset preprocessor; note that the dataset you choose must have SQuAD format                     
        └── qa_dataset.py                           # Pytorch wrapper on our dataset
    ├── trainers
        ├── __init__.py
        ├── trainer.py                              # Basic Trainer (our own wrapper around Hugging Face's Trainer)
        ├── character_based_model_trainer.py        # Trainer made for character-based models  
        └── token_based_model_trainer.py            # Trainer made for token-based models
    ├── models
        ├── __init__.py
        ├── model.py                                # Core model for Question Answering task
        ├── bert.py
        ├── mbert.py
        ├── roberta.py
        ├── xlm_roberta.py
        ├── data2vec.py
        └── canine.py         
    ├── utils  
        ├── __init__.py 
        └── utils.py                                # Bunch of utils functions
└── README.md
```

## How to run experiments:

### Install requirements

```bash
pip install -r requirements
```

Please note that you will need to update manually the versions of ``torch``, ``torchaudio`` and ``torchvision`` depending
on your hardware.

### Finetune models on SQuADv1.1 or SQuADv2 dataset:

Manually run the following commands in the **current** directory:

```bash
# To train data2vec on SQuADv2 (example)
python3 main.py \
    --model_name data2vec \
    --learning_rate 5e-5 \
    --weight_decay 1e-2 \
    --type_lr_scheduler cosine \
    --warmup_ratio 0.1 \
    --save_strategy steps \
    --save_steps 1500 \
    --num_epochs 5 \
    --early_stopping_patience 3 \
    --output_dir /mnt/hdd/dl_ensae/models \
    --device cuda \
    --batch_size 12 \
    --max_length 384 \
    --doc_stride 128 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --squad_v2 True 
    
# To train CANINE-S on SQuADv2 (example)
python3 main.py \
    --model_name canine-s \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --type_lr_scheduler constant_with_warmup\
    --warmup_ratio 0.1 \
    --save_strategy steps \
    --save_steps 1500 \
    --num_epochs 5 \
    --early_stopping_patience 3 \
    --output_dir /mnt/hdd/dl_ensae/models \
    --device cuda \
    --batch_size 6 \
    --max_length 2048 \
    --doc_stride 512 \
    --n_best_size 20 \
    --max_answer_length 256 \
    --squad_v2 True 
```

## Link to Colab

- [ ] add link

## Comparison

- [ ] data2vec
- [ ] BERT
- [ ] mBERT
- [ ] XLM-RoBERTa
- [ ] RoBERTa
- [ ] DistilBert
- [ ] CANINE

## Results \& Discussion
