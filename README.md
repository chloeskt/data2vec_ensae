# Project for the Deep Learning course at ENSAE Paris

This project has been done by Chloé SEKKAT (ENSAE \& ENS Paris-Saclay) and Mathilde KAPLOUN (ENSAE).  

The idea is to work on the first multi-modal self-supervised algorithm: [data2vec](https://arxiv.org/abs/2202.03555). 
It has been developed by FAIR quite recently and has shown promising results in speech, NLP and computer vision. Our 
goal will be to study its performance on several tasks in NLP and compare it to the best single-purpose algorithms in 
this domain. 

# Motivation and goals

Self-supervised learning is motivated by the goal of making use of large amount of unlabeled data and uncover their 
underlying structure. It is inspired by the way humans learn, namely, observation. A large part of what we learn as human 
come from us observing the world around us; not only the visual world but also sounds and language. We are constantly 
evolving thanks to our observations: there are key concepts in each objects/word/sound that we are able to grasp, 
experience on and then act on. It is our perception of the world and the associated common sense that allow humans to 
learn these basic concepts. These concepts are the building blocks of how humans learn and help us learn faster 
(think of a baby learning what "mommy" or "daddy" means).

On the opposite side, deep learning algorithms need a lot of examples and hours of training before being able to associate 
a phoneme to its written form. The goal of self-supervised learning is to design algorithms able to mimic this human 
common sense. A lot of research has been done in this field over the last 5 years, given rise to models such as BERT 
in NLP or BeiT in Computer Vision; models building on Transformers and self-attention.

However one challenge of self-supervised learning is that all developed methods are unimodal. This means that each training 
scheme/design is task-specific. For instance, there is masked language modeling for NLP and learning representations 
invariant to data augmentation in Computer Vision. Consequently, by the choice of these designs, biases are incorporated 
in each model and there is no unified way of learning. However if we think about how humans learn, we use similar 
processes for everything. This is one of the core motivation of Baevski et al. They would like to design an algorithm 
that would not be modality-specific i.e. that would benefit from cross-modal representations. They present data2vec as 
an algorithm that can learn contextualized latent representations instead of modality-specific representations. 
Concretely, the learning objective in data2vec is the same across all modalities: masked learning to produce latent 
target **continuous and contextualized representations**, using a teacher-student architecture training scheme.

# Installation 

After creating for virtual environment, please refer to each folder ``README.md`` i.e. see installation guidelines in
``source/question_answering`` for Question Answering task.

# Structure of the repository 

In the end due to time and hardware/compute constraints, we considered two mains tasks:
- question answering (F1 score \& Exact match)
- sentiment analysis ()

For more information, please refer to the corresponding `README.md` (in `source/question_answering` and `source/sentiment_classif`).

```
dl_ensae
├── .gitignore                  
├── Project_proposal            # Our project proposal
├── Final_report                # Our final report
├── source                      # Source code main package   
    ├── question_answering/     # Question answering related source code                         
    └── sentiment_classif/      # Sentiment classification related source code  
└── README.md
```

# Colab links:

For each task we developed colab notebooks that can be viewed in parallel to the source code:

- [Question Answering notebook](https://colab.research.google.com/drive/1qzDdyZ6qsNxdSMyxlCIuKqjB7FdPDw3Y?usp=sharing)
- [Sentiment Classification notebook]()
