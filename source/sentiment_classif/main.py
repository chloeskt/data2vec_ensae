import torch
import numpy as np
import torch.nn as nn
from transformers import RobertaTokenizer, Data2VecTextModel, BertTokenizer, BertModel
import random
import time
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from source.sentiment_classif.datasets import AirlineComplaints, ImdbReviews
from source.sentiment_classif.utils import EarlyStopper
from pathlib import Path

# ========================================================================


class Data2VecClassifier(nn.Module):
    def __init__(self, weights=None):
        super(Data2VecClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.tokenizer = RobertaTokenizer.from_pretrained(
            'facebook/data2vec-text-base')
        self.model = Data2VecTextModel.from_pretrained(
            'facebook/data2vec-text-base')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# ========================================================================


class BertClassifier(nn.Module):

    def __init__(self, weights=None):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# ========================================================================


class Trainer:
    # Specify loss function
    def __init__(self, model, dataset, device=torch.device("cpu"), epochs=4):
        self.set_seed()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.dataset = dataset
        self.epochs = epochs
        self.early_stopper = EarlyStopper()
        self.optimizer = AdamW(model.parameters(),
                               lr=5e-5,
                               eps=1e-8
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
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)
            t0_epoch, t0_batch = time.time(), time.time()
            total_loss, batch_loss, batch_counts = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                batch_counts += 1
                b_input_ids, b_attn_mask, b_labels = tuple(
                    t.to(self.device) for t in batch)
                self.model.zero_grad()
                logits = self.model(b_input_ids, b_attn_mask)
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                if (step % 20 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch
                    val_loss, val_accuracy = self.evaluate()
                    # print(
                    #     f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                    self.early_stopper(val_loss)
                    if self.early_stopper.early_stop:
                        print("Training complete!")
                        return (0)
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            avg_train_loss = total_loss / len(self.train_dataloader)

            print("-" * 70)
            if evaluation == True:
                val_loss, val_accuracy = self.evaluate()
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
            b_input_ids, b_attn_mask, b_labels = tuple(
                t.to(self.device) for t in batch)
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

        predictions, true_labels = [], []

        for batch in self.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask)
            logits = outputs
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.append(logits)
            true_labels.append(label_ids)
        return predictions, true_labels

    def save(self, path):
        model_path = Path('models_trained')
        torch.save(self.model.state_dict(), model_path/path)


# ========================================================================


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model = BertClassifier()
    dataset = ImdbReviews(model.tokenizer)
    trainer = Trainer(model, dataset, device=device)
    try:
        trainer.train(evaluation=True)
    except KeyboardInterrupt:
        pass
    trainer.save('bert_imdb')

    model = BertClassifier()
    dataset = AirlineComplaints(model.tokenizer)
    trainer = Trainer(model, dataset, device=device)
    try:
        trainer.train(evaluation=True)
    except KeyboardInterrupt:
        pass
    trainer.save('bert_airline')

    model = Data2VecClassifier()
    dataset = ImdbReviews(model.tokenizer)
    trainer = Trainer(model, dataset, device=device)
    try:
        trainer.train(evaluation=True)
    except KeyboardInterrupt:
        pass
    trainer.save('d2vec_imdb')

    model = Data2VecClassifier()
    dataset = AirlineComplaints(model.tokenizer)
    trainer = Trainer(model, dataset, device=device)
    try:
        trainer.train(evaluation=True)
    except KeyboardInterrupt:
        pass
    trainer.save('d2vec_airline')
