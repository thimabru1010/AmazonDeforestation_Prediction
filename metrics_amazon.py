import numpy as np
from tqdm import tqdm

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
    # true = true.reshape(-1)
    prec = Precision(pred, true)
    rec = Recall(pred, true)
    f1_clss0 = 2 * prec * rec / (prec + rec + 1e-10)
    
    # true = true.reshape(-1)
    cm, TP, FP, FN, TN = confusion_matrix(true.reshape(-1), pred.reshape(-1))
    if test_time:
        print()
        print(cm)

    prec = TN/(TN+FN + 1e-10)
    rec = TN/(TN+FP + 1e-10)
    f1_clss1 = 2 * prec * rec / (prec + rec + 1e-10)
    if np.isnan(f1_clss1):
        f1_clss1 = np.array([0])
    if np.isnan(f1_clss0):
        f1_clss0 = np.array([0])
        
    return f1_clss0, f1_clss1

def Recall(pred, true):
    # pred = np.argmax(pred, axis=2)[:, 0].reshape(-1)
    # true = true.reshape(-1)
    _, TP, _, FN, _ = confusion_matrix(true.reshape(-1), pred.reshape(-1))
    return TP/(TP+FN + 1e-10)

def Precision(pred, true):
    # pred = np.argmax(pred, axis=2)[:, 0].reshape(-1)
    # true = true.reshape(-1)
    _, TP, FP, _, _ = confusion_matrix(true.reshape(-1), pred.reshape(-1))
    return TP/(TP+FP + 1e-10)

def ACC(pred, true):
    # pred = np.argmax(pred, axis=2)[:, 0].reshape(-1)
    # true = true.reshape(-1)
    cm, TP, FP, FN, TN = confusion_matrix(true.reshape(-1), pred.reshape(-1))
    return (TP + TN) / (TP + FP + FN + TN + 1e-10)

def compute_roc(thresholds, img_predicted, img_labels):
    ''' INPUTS:
        thresholds = Vector of threshold values
        img_predicted = predicted maps (with probabilities)
        img_labels = image with labels (0-> no def, 1-> def, 2-> past def)
        mask_amazon_ts = binary tile mask (0-> train + val, 1-> test)
        px_area = not considered area (<69 pixels)

        OUTPUT:
        recall and precision for each threshold
    '''

    prec = []
    recall = []
    tpr = []
    fpr = []

    for thr in tqdm(thresholds):
        print('-'*60)
        print(f'Threshold: {thr}')

        img_predicted_ = img_predicted.copy()
        img_predicted_[img_predicted_ >= thr] = 1
        img_predicted_[img_predicted_ < thr] = 0

        ref_final = img_labels.copy()
        pre_final = img_predicted_

        # Metrics
        cm = confusion_matrix(ref_final, pre_final)

        TN = cm[0, 0]
        FP = cm[1, 0]
        TP = cm[1, 1]
        FN = cm[0, 1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)

        # TruePositiveRate = TruePositives / (TruePositives + False Negatives)
        TPR = TP / (TP + FN)
        # FalsePositiveRate = FalsePositives / (FalsePositives + TrueNegatives)
        FPR = FP / (FP + TN)

        # print(f' Precision: {precision_}')
        # print(f' Recall: {recall_}')
        print(f'TPR: {TPR}')
        print(f'FPR: {FPR}')

        tpr.append(TPR)
        fpr.append(FPR)
        prec.append(precision_)
        recall.append(recall_)

    print('-'*60)

    return prec, recall, tpr, fpr

def precision_recall_curve(thresholds, y_true, y_proba):
    # thresholds = np.unique(y_proba)
    # thresholds = np.append(thresholds, thresholds.max() + 1)  # Add a threshold higher than the maximum

    precision_values = []
    recall_values = []

    for threshold in tqdm(thresholds):
        y_pred = (y_proba >= threshold).astype(int)
        # precision = Precision(y_pred, y_true)
        # recall = Recall(y_pred, y_true)
        cm, TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
        precision = TN/(TN+FN + 1e-10)
        recall = TN/(TN+FP + 1e-10)
        # f1_clss1 = 2 * prec * rec / (prec + rec + 1e-10)
        # precision, recall, _ = calculate_metrics(y_true, y_pred)
        precision_values.append(precision)
        recall_values.append(recall)

    return np.array(precision_values), np.array(recall_values), thresholds