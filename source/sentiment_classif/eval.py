import torch
from source.sentiment_classif.main import Trainer, AirlineComplaints, Data2VecClassifier, BertClassifier
from source.sentiment_classif.imdb import ImdbReviews
import numpy as np
from sklearn import metrics

def load_model(model, path_state_dict):
    model.load_state_dict(torch.load(path_state_dict))

def eval_model(model, dataset):
    dataset.tokenize_all(model.tokenizer)
    trainer = Trainer(model, dataset)
    trainer.predict()
    return (trainer.probs)

def evaluation(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
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
    print(metrics.roc_auc_score(y_test, y_pred_prob))
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

def main():
    dataset = ImdbReviews()
    model = Data2VecClassifier()
    path = 'model_imdb'
    load_model(model, path)
    results = eval_model(model, dataset)
    save_path = "imdb_data2vec.csv"
    print(save_path)
    np.savetxt(save_path, results, delimiter=",")
