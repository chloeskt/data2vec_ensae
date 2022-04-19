from sklearn.metrics import matthews_corrcoef
import torch
from source.sentiment_classif.main import Trainer, Data2VecClassifier, BertClassifier
from source.sentiment_classif.datasets import ImdbReviews, AirlineComplaints
import numpy as np
from sklearn import metrics
from pathlib import Path


def load_model(model, path_state_dict):
    model.load_state_dict(torch.load(path_state_dict))


def eval_model(model, dataset):
    dataset.tokenize_all(model.tokenizer)
    trainer = Trainer(model, dataset)
    trainer.predict()
    return (trainer.probs)


def evaluation(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    confusion = metrics.confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print("Confusion matrix")
    print(confusion)
    print("Accuracy")
    print(accuracy)
    print("AUC score")
    print(metrics.roc_auc_score(y_true, y_pred))
    print("Matthews correlation coefficient")
    print(metrics.roc_auc_score(y_true, y_pred))
    return accuracy, confusion


def thresh(x, threshold):
    x[x < threshold] = 0
    x[x >= threshold] = 1
    return x


def eval_all(threshold=0.5):
    imdb = ImdbReviews()
    airline = AirlineComplaints()
    imdb_true = imdb.y_test
    airline_true = airline.y_test
    print(20 * '=' + "Imdb Data2vec   " + 20 * '=')
    res = np.genfromtxt('imdb_data2vec.csv', delimiter=',')
    print(res)
    # res = thresh(res, threshold)
    # evaluation(res, imdb_true)
    # print(20 * '=' + "Imdb Bert       " + 20 * '=')
    res = np.genfromtxt('imdb_bert.csv', delimiter=',')
    print(res)
    # res = thresh(res, threshold)
    # evaluation(res, imdb_true)
    # print(20 * '=' + "Airline Data2vec" + 20 * '=')
    res = np.genfromtxt('airline_data2vec.csv', delimiter=',')
    print(res)
    # res = thresh(res, threshold)
    # evaluation(res, airline_true)
    # print(20 * '=' + "Airline Bert    " + 20 * '=')
    res = np.genfromtxt('airline_bert.csv', delimiter=',')
    print(res)
    # res = thresh(res, threshold)
    # evaluation(res, airline_true)


def get_matthew_corr(predictions, true_labels):
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    return (mcc)


def main():
    model = Data2VecClassifier()
    tokenizer = model.tokenizer
    dataset = ImdbReviews(tokenizer)
    path = Path('models_trained')/'d2vec_imdb'
    load_model(model, path)
    results = eval_model(model, dataset)
    save_path = Path('predicts')/"d2vec_imdb.csv"
    print(save_path)
    np.savetxt(save_path, results, delimiter=",")
