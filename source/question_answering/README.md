# Question Answering Task

Note that we do not have access to the hyperparameters chosen by the authors of data2vec to finetune their model on 
GLUE benchmark. Moreover, we have chosen to look at F1-score and EM score instead of accuracy as we feel that it makes 
more sense.

## Organization

This subfolder contains the whole code associated with the Question Answering experiments. It can be viewed as a Python 
package whose main functions/classes can be found in the ``__init__.py``.

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

Our third experiment consists in measuring the abilities of data2vec to adapt to new target domain by only doing few-shot 
learning. This means that we want to take a finetuned data2vec model (on SQuADv2 which is a general wikipedia-based dataset) 
and measure its performance on another domain-specific dataset (for instance medical or legal datasets which are two 
domains with very specific wording and concepts) after having train it for a small number of epochs (3 or less) on a very 
small number of labeled data (less than 250 for instance). These performances will be compared to those of the other 
models we have chosen along this study. 

Last, we will stay again in the few-shot learning domain but test the abilities of data2vec to resist to adversarial 
attacks knowing that it has not been trained for that and that it will only be trained for few epochs and a small number 
of adversarial examples. 

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
    --output_dir <<CHOSEN_DIRECTORY>> \
    --device cuda \
    --batch_size 12 \
    --max_length 384 \
    --doc_stride 128 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --squad_v2 True \
    
# To train CANINE-S on SQuADv2 (example)
python3 main.py \
    --model_name canine-s \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --type_lr_scheduler linear \
    --warmup_ratio 0.1 \
    --save_strategy steps \
    --save_steps 1500 \
    --num_epochs 5 \
    --early_stopping_patience 3 \
    --output_dir <<CHOSEN_DIRECTORY>> \
    --device cuda \
    --batch_size 6 \
    --max_length 2048 \
    --doc_stride 512 \
    --n_best_size 20 \
    --max_answer_length 256 \
    --squad_v2 True \
```

## Link to Colab

To see more comments and results, please have a look at the following [Colab](https://colab.research.google.com/drive/1qzDdyZ6qsNxdSMyxlCIuKqjB7FdPDw3Y?usp=sharing).

## Link to finetuned models and datasets

Finetuned models used in our experiences and custom datasets are available [here](https://drive.google.com/drive/folders/1L9Su25qatgdmoz-rZbeY_tA2bXq9T9EG?usp=sharing).

## Results \& Discussion

### Extractive Question Answering on SQuADv2

Models are finetuned on SQuADv2 training set with the following parameters:

|             	| Batch size 	| Learning Rate 	| Weigh decay 	| Nb of epochs 	| Number of training examples 	| Number of validation examples 	| Max sequence length 	| Doc stride 	| Max answer length 	| Lr scheduler 	| Warmup ratio 	|
|:-----------:	|:----------:	|:-------------:	|:-----------:	|:------------:	|:---------------------------:	|:-----------------------------:	|:-------------------:	|:----------:	|:-----------------:	|:------------:	|:------------:	|
|   data2vec  	|     12     	|      2e-5     	|     1e-4    	|       3      	|            131823           	|             12165             	|         348         	|     128    	|         30        	|    cosine    	|      0.1     	|
|   RoBERTa   	|     12     	|      2e-5     	|     1e-4    	|       3      	|            131823           	|             12165             	|         348         	|     128    	|         30        	|    cosine    	|      0.1     	|
|     BERT    	|      8     	|      3e-5     	|      0      	|              	|            131754           	|             12134             	|         348         	|     128    	|         30        	|    linear    	|       0      	|
|  DistilBERT 	|      8     	|      3e-5     	|     1e-2    	|       2      	|            131754           	|             12134             	|         348         	|     128    	|         30        	|    linear    	|      0.1     	|
|    mBERT    	|      8     	|      2e-5     	|      0      	|       2      	|            132335           	|             12245             	|         348         	|     128    	|         30        	|    linear    	|       0      	|
| XLM-ROBERTA 	|      8     	|      3e-5     	|      0      	|       2      	|            133317           	|             12360             	|         348         	|     128    	|         30        	|    linear    	|       0      	|
|   CANINE-c  	|      4     	|      5e-5     	|     0.01    	|       3      	|            130303           	|             11861             	|         2048        	|     512    	|        256        	|    linear    	|      0.1     	|
|   CANINE-s  	|      4     	|      5e-5     	|    0.001    	|      2.5     	|            130303           	|             11861             	|         2048        	|     512    	|        256        	|    linear    	|      0.1     	|


Results are given in the following tables:

|                 	| **F1-score** 	| **EM score** 	|
|:---------------:	|:------------:	|:------------:	|
|   **data2vec**  	|     78.21    	|     74.90    	|
|     **BERT**    	|     76.74    	|     73.59    	|
|   **RoBERTa**   	|     82.02    	|     78.54    	|
|  **DistilBERT** 	|     67.81    	|     64.71    	|
|   **CANINE-C**  	|     74.1     	|     69.2     	|
|   **CANINE-S**  	|     72.5     	|     69.6     	|
|    **mBERT**    	|     77.51    	|     74.1     	|
| **XLM-RoBERTa** 	|     78.3     	|     75.12    	|

data2vec performs decently well, reaching a F1 score of $78.21$, right after RoBERTa ($82.02$). The former outperforms 
every other models by at least one point. What is noticeable is that multi-lingual token-based models perform better than 
unilingual ones on English dataset. This hints that cross-lingual features can help to get more accurate learning 
representation of words. data2vec performs even better than that. It might be due to the contextualized embeddings which 
contains higher level information. RoBERTa is even better. Our intuition is that this is due to the masking scheme applied, 
which differs from the one applied in BERT. In the former, the masking is done during training, the authors generate masks 
on the sequence when it gets fed to the model. This means that the number of combinations of masked versions for a given 
sentence is way larger than the fixed number in BERT masking strategy.

### Robustness to noise

We created a noisy version of the SQuADv2 dataset where, given a noise level $p$, we transform each word into a noisy 
version thanks to the [nlpaug library](https://github.com/makcedward/nlpaug). We propose three levels of noise: $10$\%, 
$20\%$ and $40\%$. We propose 4 types of noise: _KeyboardAug_ (mimics error due to the keyboard, substitutes 
character by keyboard distance), _RandomCharAug_ (insert, swap, substitute or delete a character), _OcrAug_ 
(substitute a character by pre-defined OCR error) and _BackTranslationAug_ (from English to German, then German 
to English).

For compute reasons, we only tested the _RandomCharAug_-substitution noise. The noise is applied on the SQuADv2 
test set while the models are finetuned on a clean version of the training set (we took the finetuned models obtained 
in the previous subsection). Results are given in the following table:

|                 	| **Noise level 10%** 	|        	| **Noise level 20%** 	|        	| **Noise level 40%** 	|        	|
|:---------------:	|:-------------------:	|:------:	|:-------------------:	|:------:	|:-------------------:	|:------:	|
|                 	|     **F1 score**    	| **EM** 	|     **F1 score**    	| **EM** 	|     **F1 score**    	| **EM** 	|
|   **data2vec**  	|         75,1        	|  71,8  	|        72,53        	|  69,05 	|        67,14        	|  64,68 	|
|     **BERT**    	|        73,68        	|  70,79 	|        71,22        	|  68,55 	|        66,42        	|  63,74 	|
|   **RoBERTa**   	|        79,06        	|  75,87 	|        76,57        	|  73,56 	|         70,7        	|  68,18 	|
|  **DistilBERT** 	|        65,85        	|  63,05 	|        64,42        	|  61,92 	|        60,77        	|  58,78 	|
|    **mBERT**    	|          74         	|  70,75 	|        71,66        	|  68,46 	|        67,08        	|  64,74 	|
| **XLM-RoBERTa** 	|        74,54        	|  71,61 	|        72,68        	|  69,81 	|        67,12        	|  64,43 	|
|   **CANINE-C**  	|        69,64        	|  66,89 	|        67,88        	|  65,43 	|        66,03        	|  63,9  	|
|   **CANINE-S**  	|        72,25        	|  69,65 	|         70,3        	|  68,03 	|        67,18        	|  64,6  	|

Again RoBERTa seem to be the best model at hand. It is the most robust across the three noise levels. data2vec also 
appears to be quite robust compared to other models, it is often second or third in line. When $p=20$\%, it is quite close 
to the performances of XLM-RoBERTa (while having less parameters). One might notice that as the level of noise increases, 
CANINE-S model actually reaches the second best score (close to data2vec, $67.18$ vs. $67.14$). This is quite interesting.
It hints that tokenization-free models might be better to use in presence of noisy input (or out-of-vocabulary words since 
it is not bounded to a fixed vocabulary; the latter is a constraint of token-based models only). Moreover, this experience 
also highlights that the latent representations provided by data2vec are more robust to this keyboard-type of noise, 
regardless of the noise level. This makes it very competitive in real-life settings and also is a step closer to generic 
understanding and imitation of human capacities as we, humans, are able to understand the vast majority of a text even 
when it is noisy, especially when it is keyboard-type noise and random substitution of characters.

### Few-shot learning and domain adaptation

The goal of this experiment is to measure the ability of data2vec (and other models) to transfer to unseen data, in 
another domain. This could either be done in zero-shot or few-shot settings. Here we decided to go with the latter as it 
is more realistic. In real life, a company might already have a custom small database of labeled documents and questions 
associated (manually created) but would want to deploy a Question Answering system on the whole unlabeled database. The 
CUAD dataset is perfect for this task as it is highly specialized (legal domain, legal contract review). The training set 
is made of 22450 question/context pairs and the test set of 4182. We randomly selected 1\% of the training set (224 examples) 
to train on for 3 epochs, using the previosuly finetuned models on SQuADv2. Then each model was evaluated on 656 test examples. 
Results are reported in the following table and to ensure fair comparison, all models where trained and tested on the 
exact same examples. 

|                 	| **F1 score** 	| **EM score** 	|
|:---------------:	|:------------:	|:------------:	|
|   **data2vec**  	|     74.57    	|     72.24    	|
|     **BERT**    	|     74.18    	|     72.72    	|
|   **RoBERTa**   	|     73.83    	|     72.24    	|
|  **DistilBERT** 	|     72.86    	|     71.37    	|
|    **mBERT**    	|     74.50    	|     73.12    	|
| **XLM-RoBERTa** 	|     76.64    	|     73.44    	|
|   **CANINE-C**  	|     72.51    	|     71.39    	|
|   **CANINE-S**  	|     72.27    	|     71.27    	|

All models perform quite similarly except for XLM-RoBERTa which has + 2F1 compared to data2vec, second-best model in line. 
This experiment highlights the quality of the latent representation produced by data2vec and their ability to quickly 
adapt to new domain. However, we were not able to see a great difference with the abilities of other BERT-like models. 

### Few-shot learning and adversarial attacks

This last Question Answering-related experiment aims at testing data2vec abilities not to be fooled in adversarial settings. 
We decided to us the  dynabench/QA dataset (BERT-version). The latter is an adversarially collected Reading Comprehension 
dataset spanning over multiple rounds of data collect. It has been made so that SOTA NLP models find it challenging. 

We decided to take models finetuned on SQuADv2, take 200 examples (2\%) extracted from dynabench/qa training set to train 
each model for 3 epochs and then evaluate these models on 600 test examples (60\% of the full test set).Our results are 
displayed in the following table. Again, to ensure fair comparison, all models are trained on the exact same examples 
and evaluated on the same ones.

|                 	| **F1 score** 	| **EM score** 	|
|:---------------:	|:------------:	|:------------:	|
|   **data2vec**  	|   **39.33**  	|     29.6     	|
|     **BERT**    	|     38.13    	|     25.6     	|
|   **RoBERTa**   	|   **47.47**  	|     35.8     	|
|  **DistilBERT** 	|     32.64    	|     22.5     	|
|    **mBERT**    	|     38.43    	|     28.6     	|
| **XLM-RoBERTa** 	|     36.51    	|     27.6     	|
|   **CANINE-C**  	|     28.25    	|     18.6     	|
|   **CANINE-S**  	|     27.40    	|     17.2     	|

RoBERTa is the model with the best performance, with more than 8F1 points then the second-best in line, data2vec. Again, 
data2vec shows great abilities compared to BERT-like models but is never the first choice in terms of performance, either 
RoBERTa or XLM-RoBERTa model is often better. However, one might note that data2vec is smaller in terms of parameters 
hence faster to train (especially compared to XLM-RoBERTa which is known to be a huge model). This makes data2vec attractive. 

Finally, we observed that CANINE models are much more prone to adversarial attacks (-10F1 points compared to data2vec and 
BERT). It is yet unclear for us why it is the case. Surely this is due to the fact that CANINE is tokenization-free but, 
we still need to build intuition on why this has a great impact when evaluated on adversarial samples.
