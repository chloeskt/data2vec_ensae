# Question Answering Task

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It can developed by Chloé 
Sekkat and can be viewed as a Python package whose main functions/classes can be found in the ``__init__.py``.

## Task description

## Structure

## How to run experiments:

## Install requirements

```bash
pip install -r requirements
```

Please note that you will need to update manually the versions of ``torch``, ``torchaudio`` and ``torchvision`` depending
on your hardware.

### Finetune models on SQuADv1.1 or SQuADv2 dataset:

Manually run the following commands in the **current** directory:

```bash
# To train data2vec on SQuADv2
python3 main.py \
    --model_name data2vec \
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
    --batch_size 12 \
    --max_length 384 \
    --doc_stride 128 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --squad_v2 True 
    
# To train CANINE-S on SQuADv2
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

## Comparison

- [ ] data2vec
-  [ ] BERT
-  [ ] mBERT
-  [ ] XLM-RoBERTa
-  [ ] CANINE

## Results \& Discussion
