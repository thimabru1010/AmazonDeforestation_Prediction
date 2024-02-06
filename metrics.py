import numpy as np

def confusion_matrix(true, pred, num_classes=2):
    # Inicialize uma matriz de confusão (confusion matrix) como uma matriz zeros 3x3
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Preencha a matriz de confusão
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((true == i) & (pred == j))
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[1, 0]
    FN = confusion_matrix[0, 1]
    TN = confusion_matrix[1, 1]
    return confusion_matrix, TP, FP, FN, TN

def f1_score(pred, true, test_time=False):
    _pred = (pred >= 0.5).reshape(-1)
    _true = true.reshape(-1)
    prec = Precision(pred, true)
    rec = Recall(pred, true)
    f1_clss0 = 2 * prec * rec / (prec + rec)

    _true = true.reshape(-1)
    cm, TP, FP, FN, TN = confusion_matrix(_true, _pred)
    if test_time:
        print()
        print(cm)
    prec = TN/(TN+FN)
    rec = TN/(TN+FP)
    f1_clss1 = 2 * prec * rec / (prec + rec)
    if np.isnan(f1_clss1):
        f1_clss1 = np.array([0])
    if np.isnan(f1_clss0):
        f1_clss0 = np.array([0])
    return f1_clss0, f1_clss1

def Recall(pred, true):
    pred = (pred >= 0.5).reshape(-1)
    true = true.reshape(-1)
    _, TP, _, FN, _ = confusion_matrix(true, pred)
    return TP/(TP+FN)

def Precision(pred, true):
    pred = (pred >= 0.5).reshape(-1)
    true = true.reshape(-1)
    _, TP, FP, _, _ = confusion_matrix(true, pred)
    return TP/(TP+FP)

def ACC(pred, true):
    pred = (pred >= 0.5).reshape(-1)
    true = true.reshape(-1)
    cm, TP, FP, FN, TN = confusion_matrix(true, pred)
    return (TP + TN) / (TP + FP + FN + TN)