import os
import requests
import zipfile
import pandas as pd
import tarfile
from pathlib import Path
from source.sentiment_classif.main import Dataset_
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import re
import requests, zipfile
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, Data2VecTextModel, BertTokenizer, BertModel
import random
import time
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

class ImdbReviews(Dataset_):
    def __init__(self, split_ratios=(0.75, 0.15, 0.10), batch_size=32, max_len=200):
        self.split_ratios = split_ratios
        self.dataset = self.get_data()
        self.format()
        self.X = self.dataset.X
        self.y = self.dataset.y
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = self.split()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_labels = torch.tensor(self.y_train)
        self.val_labels = torch.tensor(self.y_val)

    def split(self):
        train_ratio, validation_ratio, test_ratio = self.split_ratios
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1 - train_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
        return (x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy(), x_val.to_numpy(), y_val.to_numpy())

    @staticmethod
    def get_data():
        zip_file = Path('imdb.zip')
        data_dir = Path('./imdb')
        data_file = Path('imdb.csv')

        if not zip_file.is_file() and not data_file.is_file():
            raise Exception("Dataset not uploaded")

        if not data_file.is_file():
            print("Extracting data ...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.rename(data_dir/'IMDB Dataset.csv', data_file)
            os.rmdir(data_dir)
        if zip_file.is_file():
            os.remove(zip_file)
        return (pd.read_csv(data_file))

    def format(self):
        self.dataset = self.dataset.rename(columns={'review': 'X', 'sentiment': 'y'})
        self.dataset.y = self.dataset.y.map({'negative': 0, 'positive' :1})

    def sample(self):
        self.data.sample(5)

    def sample_test(self):
        self.test.sample(5)

    def __len__(self):
        return len(self.data)

    def tokenization(self, data, tokenizer):
        input_ids = []
        attention_masks = []

        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=self.preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                return_attention_mask=True  # Return attention mask
            )

            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def tokenize_all(self, tokenizer):
        print("Tokenizing")
        train_inputs, train_masks = self.tokenization(self.X_train, tokenizer)
        val_inputs, val_masks = self.tokenization(self.X_val, tokenizer)
        test_inputs, test_masks = self.tokenization(self.X_test, tokenizer)
        self.train_data = TensorDataset(train_inputs, train_masks, self.train_labels)
        self.train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=self.batch_size)
        self.val_data = TensorDataset(val_inputs, val_masks, self.val_labels)
        self.val_sampler = SequentialSampler(self.val_data)
        self.val_dataloader = DataLoader(self.val_data, sampler=self.val_sampler, batch_size=self.batch_size)
        self.test_data = TensorDataset(test_inputs, test_masks)
        self.test_sampler = SequentialSampler(self.test_data)
        self.test_dataloader = DataLoader(self.test_data, sampler=self.test_sampler, batch_size=self.batch_size)

    def get_max_len(self, tokenizer):
        all_data = np.concatenate([self.X, self.X_test.values])
        encoded_data = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_data]
        max_len = max([len(sent) for sent in encoded_data])
        return (max_len)

def main():
    dataset = ImdbReviews()
    print(dataset.dataset.sample(5))
