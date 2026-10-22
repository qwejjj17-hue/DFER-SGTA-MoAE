import sklearn.metrics


def get_WAR(trues, preds):
    """Weighted Average Recall, equivalent to overall accuracy."""
    return sklearn.metrics.accuracy_score(trues, preds)


def get_UAR(trues, preds):
    """Unweighted Average Recall: mean per-class recall."""
    cm = sklearn.metrics.confusion_matrix(trues, preds)
    acc_per_cls = [cm[i, i] / sum(cm[i]) for i in range(len(cm))]
    return sum(acc_per_cls) / len(acc_per_cls)


def get_cm(trues, preds):
    """Confusion matrix."""
    return sklearn.metrics.confusion_matrix(trues, preds)
