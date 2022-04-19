import pandas as pd
from pathlib import Path
import os
import gdown
import re
import emoji
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

TWEET_LINK = 'https://drive.google.com/uc?id=1oaFiTnHQEkrtdSX79nhN4Rk0g3ptFarE'
IMDB_LINK = 'https://drive.google.com/uc?id=1XfhS88C8I8u4TZTs7mpujJl0KDYD-ZsM'


class SentimentDataset:
    def __init__(self, tokenizer, split_ratios=(0.80, 0.10), batch_size=32, max_len=175):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.data_dir = Path('datasets')

    def get_ratios(self):
        train_ratio, val_ratio = self.split_ratios
        len_df = len(self.dataset)
        train_size = int(len_df * train_ratio)
        val_size = int(len_df * val_ratio)
        test_size = len_df - train_size - val_size
        return train_size, val_size, test_size

    def get_dataloaders(self):
        train_dataloader = DataLoader(
            self.train_df,
            sampler=RandomSampler(self.train_df),
            batch_size=self.batch_size
        )
        validation_dataloader = DataLoader(
            self.val_df,
            sampler=SequentialSampler(self.val_df),
            batch_size=self.batch_size
        )
        test_dataloader = DataLoader(
            self.test_df,
            sampler=SequentialSampler(self.test_df),
            batch_size=self.batch_size
        )
        return train_dataloader, validation_dataloader, test_dataloader

    def download(self):
        if not self.data_dir.exists():
            os.mkdir(self.data_dir)
        if not self.path.is_file():
            print("Downloading ...")
            with open(self.path, 'wb') as f:
                gdown.download(self.url, f, quiet=False, fuzzy=True)
        else:
            print('Dataset already downloaded')

    def sample(self):
        return self.df.sample(5)

    def __len__(self):
        return len(self.df)

    def tokenize(self):
        input_ids = []
        attention_masks = []

        for sentence in self.df.X.values:
            encoded_sent = self.tokenizer.encode_plus(
                text=sentence,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                return_attention_mask=True  # Return attention mask
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        dataset = TensorDataset(
            input_ids, attention_masks, torch.tensor(self.df.y.values))
        return dataset

    def get_max_len(self, tokenizer):
        max_len = 0
        for sentence in self.data.X:
            input_ids = tokenizer.encode(sentence, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        return (max_len)


class AirlineComplaints(SentimentDataset):
    def __init__(self, tokenizer, split_ratios=(0.80, 0.10), batch_size=32, max_len=175):
        super().__init__(
            tokenizer, split_ratios, batch_size, max_len)
        self.url = TWEET_LINK
        self.path = self.data_dir/Path('airline_data.csv')
        self.download()
        self.df = pd.read_csv(self.path)
        self.format()
        self.dataset = self.tokenize()
        self.split = self.get_ratios()
        self.train_df, self.val_df, self.test_df = random_split(
            self.dataset, self.split)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_dataloaders()

    def format(self):
        print("Formatting data ...")
        self.df = self.df[['airline_sentiment', 'text']]
        self.df = self.df.rename(
            columns={'airline_sentiment': 'y', 'text': 'X'})
        self.df = self.df[self.df.y != 'neutral']
        self.df.X = self.df.X.apply(self.preprocessing)
        self.df.y = self.df.y.apply(lambda s: 0 if s == 'negative' else 1)

    def preprocessing(self, text):
        text = emoji.replace_emoji(text)  # Strip emojis
        text = re.sub(r'(@.*?)[\s]', ' ', text)  # Remove mentions
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'\s+', ' ', text).strip()  # Remove trailing whitespaces
        return (text)


class ImdbReviews(SentimentDataset):
    def __init__(self, tokenizer, split_ratios=(0.80, 0.10), batch_size=32, max_len=200):
        super().__init__(
            tokenizer, split_ratios, batch_size, max_len)
        self.url = IMDB_LINK
        self.path = self.data_dir/Path('imdb_data.csv')
        self.download()
        self.df = pd.read_csv(self.path)
        self.format()
        self.dataset = self.tokenize()
        self.split = self.get_ratios()
        self.train_df, self.val_df, self.test_df = random_split(
            self.dataset, self.split)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_dataloaders()

    def format(self):  # Changed
        print("Formatting data ...")
        self.df = self.df.rename(
            columns={'sentiment': 'y', 'review': 'X'})
        self.df.X = self.df.X.apply(self.preprocessing)
        self.df.y = self.df.y.apply(lambda s: 0 if s == 'negative' else 1)
        self.df = self.df.sample(10000)

    def preprocessing(self, text):  # Changed
        text = re.sub(r'\s+', ' ', text).strip()  # Remove trailing whitespaces
        test = re.sub(r'<[^>]+>', ' ', text)  # Remove html tagsh
        return (text)
