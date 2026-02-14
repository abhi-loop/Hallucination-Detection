from sklearn.metrics import roc_auc_score, accuracy_score

def compute_auroc(scores, labels):
    return roc_auc_score(labels, scores)

def classify(scores, threshold):
    return [1 if s > threshold else 0 for s in scores]

def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds)
