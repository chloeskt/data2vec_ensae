# Question Answering Task

Note that we do not have access to the hyperparameters chosen by the authors of data2vec to finetune their model on 
GLUE benchmark. Moreover, we have chosen to look at F1-score and EM score instead of accuracy as we feel that it makes 
more sense.

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It has been developed by Chloé 
Sekkat and can be viewed as a Python package whose main functions/classes can be found in the ``__init__.py``.

## Task description

In this section, we are interested by the capacities of the embeddings produced by data2vec against both token-based models
such as BERT, RoBERTa, mBERT and XLM-RoBERTa and token-free models (character-based) such as CANINE on Question Answering tasks.
Note CANINE is a pre-trained tokenization-free and vocabulary-free encoder, that operates directly on character sequences without explicit
tokenization. It seeks to generalize beyond the orthographic forms encountered during pre-training. data2vec is an 
algorithm that can learn a contextualized latent representations instead of modality-specific representations. Its 
learning objective is the same across all modalities: masked learning to produce latent target **continuous and 
contextualized representations**, using a teacher-student architecture training scheme.

We evaluate its capacities on extractive question answering (select minimal span answer within a context) on SQuAD dataset. The latter is
a unilingual (English) dataset available in Hugging Face (simple as ``load_dataset("squad_v2")``). Obtained F1-scores are being
compared to BERT-like models (BERT, mBERT and XLM-RoBERTa) and CANINE. Note that mBERT, XLM-RoBERTa and CANINE where
trained on **multilingual data**. **Therefore data2vec is only directly comparable to BERT and RoBERTa, but it is still interesting to
get the performances on other models on the same task.**

A second step is to assess its capacities of generalization in the context of zero-shot transfer. Finetuned on an English
dataset and then directly evaluated on a multi-lingual dataset with 11 languages of various morphologies (XQuAD). Intuitively, 
only the models trained on multilingual data will be able to generalize. Therefore this experience will only allow us to
compare mBERT, XLM-RoBERTa and CANINE. 

A third step is to assess data2vec's capacities to handle noisy inputs, especially noisy questions as it is highly 
probable that in real life settings, the ASR system or the human typing the question will not provide a perfect/clean
question [TODO]. 

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
