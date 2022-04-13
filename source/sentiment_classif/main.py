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

class Dataset_:
    def __init__(self):
        self.data = pd.DataFrame()
        self.test = pd.DataFrame()

    def download(self):
        pass

    def unzip(self):
        pass

    def format(self):
        pass

    def preprocessing(self, text):
        pass

    def sample(self):
        self.data.sample(5)

    def sample_test(self):
        self.test.sample(5)

    def __len__(self):
        return len(self.data)

    def __get__(self):
        return self.data

    def tokenization(self, data, tokenizer):
        preprocessing = dataset.preprocessing
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

class AirlineComplaints(Dataset_):
    def __init__(self, test_size=0.1, batch_size=32, max_len=175):
        self.url = 'https://drive.google.com/uc?export=download&id=1wHt8PsMLsfX5yNSqrt2fSTcb8LEiclcf'
        self.download()
        self.unzip()
        self.format()
        self.X = self.data.tweet.values
        self.y = self.data.label.values
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        self.train_labels = torch.tensor(self.y_train)
        self.val_labels = torch.tensor(self.y_val)
        self.batch_size = batch_size
        self.max_len = max_len
        self.X_test = self.test.tweet

    def download(self):
        print("Downloading ...")
        data = requests.get(self.url)
        with open('data.zip', 'wb') as f:
            f.write(data.content)

    def unzip(self):
        print("Extracting ...")
        with zipfile.ZipFile('data.zip') as z:
            z.extractall('data')

    def format(self):
        print("Formatting data ...")
        complaints = pd.read_csv('data/complaint1700.csv')
        non_complaints = pd.read_csv('data/noncomplaint1700.csv')
        complaints['label'] = 0
        non_complaints['label'] = 1
        self.data = pd.concat([complaints, non_complaints], axis=0).reset_index(drop=True)
        self.data = self.data.drop(['airline'], axis=1)
        self.test = pd.read_csv('data/test_data.csv')
        self.test = self.test[['id', 'tweet']]

    def preprocessing(self, text):
        text = re.sub(r'(@.*?)[\s]', ' ', text)  # Remove mentions
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'\s+', ' ', text).strip()  # Remove trailing whitespaces
        return (text)

    def sample(self):
        self.data.sample(5)

    def sample_test(self):
        self.test.sample(5)

    def __len__(self):
        return len(self.data)

    def tokenization(self, data, tokenizer):
        preprocessing = dataset.preprocessing
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
        all_tweets = np.concatenate([self.X, self.X_test.values])
        encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]
        max_len = max([len(sent) for sent in encoded_tweets])
        return (max_len)

# ========================================================================

class Data2VecClassifier(nn.Module):
    def __init__(self):
        super(Data2VecClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.tokenizer = RobertaTokenizer.from_pretrained('facebook/data2vec-text-base')
        self.model = Data2VecTextModel.from_pretrained('facebook/data2vec-text-base')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# ========================================================================

class BertClassifier(nn.Module):

    def __init__(self):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# ========================================================================

class Trainer:
    # Specify loss function
    def __init__(self, model, dataset, device=torch.device("cuda"), epochs=4):
        self.set_seed()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.model.to(device)
        self.dataset = dataset
        self.epochs = epochs
        self.device = device
        self.optimizer = AdamW(model.parameters(),
                               lr=5e-5,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )
        self.steps = len(dataset) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value
                                                         num_training_steps=self.steps)
        self.train_dataloader = self.dataset.train_dataloader
        self.val_dataloader = self.dataset.val_dataloader
        self.test_dataloader = self.dataset.test_dataloader

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def train(self, evaluation=False):
        print("Start training...\n")
        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(self.train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.evaluate()

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 70)
            print("\n")

        print("Training complete!")

    def evaluate(self):
        self.model.eval()

        val_accuracy = []
        val_loss = []

        for batch in self.val_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        return val_loss, val_accuracy

    def predict(self):
        self.model.eval()

        all_logits = []
        for batch in self.test_dataloader:
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        self.probs = F.softmax(all_logits, dim=1).cpu().numpy()

    def evaluate_roc(self):
        preds = self.probs[:, 1]
        fpr, tpr, threshold = roc_curve(self.dataset.y_val, preds)
        roc_auc = auc(fpr, tpr)
        print(f'AUC: {roc_auc:.4f}')

        # Get accuracy over the test set
        y_pred = np.where(preds >= 0.5, 1, 0)
        accuracy = accuracy_score(self.dataset.y_val, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Plot ROC AUC
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

# ========================================================================

if __name__ == '__main__':

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using GPU")
    # else:
    #     print('No GPU available, using the CPU instead.')
    #     device = torch.device("cpu")

    dataset = AirlineComplaints()
    # model = Data2VecClassifier()
    # dataset.tokenize_all(model.tokenizer)
    # trainer = Trainer(model, dataset, device=device)
    # trainer.train(evaluation=True)
