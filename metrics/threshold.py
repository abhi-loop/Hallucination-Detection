import numpy as np
from sklearn.metrics import roc_curve, auc

def find_best_threshold(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmeans)
    return thresholds[idx], gmeans[idx], auc(fpr, tpr)