# Project for the Deep Learning course at ENSAE Paris

This project has been done by Chloé SEKKAT (ENSAE \& ENS Paris-Saclay) and Mathilde KAPLOUN (ENSAE).  

The idea is to work on the first multi-modal self-supervised algorithm: [data2vec](https://arxiv.org/abs/2202.03555). 
It has been developed by FAIR quite recently and has shown promising results in speech, NLP and computer vision. Our 
goal will be to study its performance on several tasks in NLP and compare it to the best single-purpose algorithms in 
this domain. 

# Motivation and goal

## Problem definition 

One challenge of self-supervised learning is that all developed methods are unimodal. This means that each scheme is 
task-specific, e.g. Masked Language Model for NLP, learning representations invariant to data augmentation in Computer 
Vision. Consequently, attached to each task, are biased in their own way. [Baevski et al.](https://arxiv.org/abs/2202.03555) 
are motivated by the core idea that humans use similar processes to interacted and understand the visual world, language 
and sound. Is it possible to design an algorithm inspired by such processes ? An algorithm that can learn a contextualized 
latent representations instead of modality-specific representations. Concretely, the learning objective in _data2vec_ is 
the same across all modalities: masked learning to produce latent target _continuous and contextualized representations_, 
using a teacher-student architecture training scheme.

## Models

First, we will make sure to understand how _data2vec_ work and differ from other self-supervised models in NLP and 
Computer Vision. Then, the \textsc{data2vec} will be compared against SOTA models such as BERT or RoBERTa in NLP, and 
ViT in Computer Vision on a various set of downstream tasks. Results provided by the paper suggest that _data2vec_ 
overperforms such models. It would be nice (if pre-trained models are available) is to compare _data2vec_ with DINO 
since both of them use the teacher-student architecture but differ in the prediction task. 

Please note that currently pretrained models for Computer Vision are **not** [available](https://github.com/pytorch/fairseq/tree/main/examples/data2vec). 
Depending on when FAIR will make one available, we might **not** be able to evaluate data2vec on downstream Computer Vision 
tasks. Moreover, note that currently fine-runed pre-trained models on NLP tasks are **not** available, which means that 
we will fine-tune ourselves the base model. This will allow us to get a better understanding on how to fine-tune such 
models and will make us dive into the source code. However this also implies that we shall make experiments in low budget
computational settings as we do not have access to huge clusters as FAIR does.

## Datasets

- Computer Vision: as we will try to first reproduce the paper's results, we will use the well-known benchmark ImageNet, 
on the downstream tasks of image classification. Other tasks might be considered such as image captioning using the [COCO 
(Common Objects in Context) dataset](https://cocodataset.org/).
- Natural Language Processing: in order to reproduce the paper's results, we shall fine-tune the model on the GLUe benchmark. 
Note that not all tasks might be considered e.g. we might consider only Question Answering (SQUAD), Natural Language Inference 
(MNLI) and Sentiment Analysis (SST-2). One interesting thing would be to evaluate data2vec performances on multi-lingual 
Question Answering for instance (using XQUAD in zero-shot transfer).

## Evaluation

For image classification we will focus on top-1 validation accuracy on ImageNet-1K as it is a standard benchmark. 
For NLP, depending on the downstream task we will look at the F1-score (Question Answering), accuracy on both the macthed 
and unmatched dev sets (Natural Language Inference) and the unweighted average of Pearson and Spearman correlation or 
accuracy (Sentiment Analysis). 

# Installation 

After creating for virtual environment, please run:

``
pip install -r requirements
``

Please note that you will need to update manually the versions of torch, torchaudio and torchvision depending on your hardware.

# Structure of the repository 

- [ ] TO BE DONE
