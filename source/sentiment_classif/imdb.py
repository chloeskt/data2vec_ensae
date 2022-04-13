import os
import requests
import zipfile
import pandas as pd
import tarfile
from pathlib import Path
from main import Dataset_
from sklearn.model_selection import train_test_split

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

    def split(self):
        train_ratio, validation_ratio, test_ratio = self.split_ratios
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1 - train_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
        return (x_train, y_train, x_test, y_test, x_val, y_val)

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

def main():
    dataset = ImdbReviews()
    print(dataset.dataset.sample(5))
