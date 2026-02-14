import numpy as np
from sklearn.metrics import roc_curve

def find_best_threshold(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)

    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)

    return thresholds[ix], gmeans[ix]
