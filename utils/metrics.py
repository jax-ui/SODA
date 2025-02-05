
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
from imblearn.metrics import geometric_mean_score

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
  
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    metrics = {}
        # 1. Average Precision
    if len(np.unique(labels)) > 1:
        try:
            average_precision = average_precision_score(y_true=labels, y_score=predicts)
        except ValueError:
            average_precision = np.nan
    else:
        average_precision = np.nan
    metrics['average_precision'] = average_precision * 100

    # 2. ROC AUC
    if len(np.unique(labels)) > 1:
        try:
            roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        except ValueError:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
    metrics['roc_auc'] = roc_auc * 100

    # 3. F1 Score
    threshold = 0.3
    binary_predicts = (predicts >= threshold).astype(int)
    if len(np.unique(binary_predicts)) > 1:  # Ensure both classes are present
        f1 = f1_score(y_true=labels, y_pred=binary_predicts)
    else:
        f1 = 0.0  # No positive predictions or imbalanced predictions
    metrics['f1'] = f1 * 100

    # 4. G-Mean
    
    # g_mean = geometric_mean_score(labels, binary_predicts, average='micro')
    # metrics['g-mean'] = g_mean * 100
    # #  average precision
    # if len(np.unique(labels)) > 1:
    #     average_precision = average_precision_score(y_true=labels, y_score=predicts)
    # else:
    #     average_precision = np.nan
    # metrics['average_precision'] = average_precision

    # #  ROC AUC
    # if len(np.unique(labels)) > 1:
    #     roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    # else:
    #     roc_auc = np.nan
    # metrics['roc_auc'] = roc_auc

    # # F1 score
    # f1 = f1_score(y_true=labels, y_pred=(predicts > 0.5).astype(int))
    # metrics['f1'] = f1
    # #G-mean
    # threshold = 0.5
    # binary_predicts = (predicts >= threshold).astype(int)
    # cm = confusion_matrix(labels, binary_predicts)
    # if cm.shape == (1, 1):
    #     if labels[0] == 0:
    #         tn, fp, fn, tp = cm[0, 0], 0, 0, 0
    #     else:
    #         tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    # else:
    #     tn, fp, fn, tp = cm.ravel()

    # def get_sensitivity(tp, fn):
    #     if tp + fn == 0:
    #         return 0.0 
    #     else:
    #         return tp / (tp + fn)


    # sensitivity = get_sensitivity(tp, fn)
    # specificity = tn / (tn + fp)
    # g_mean = np.sqrt(sensitivity * specificity)
    # metrics['g-mean'] = g_mean

    return metrics

    # return {'average_precision': average_precision, 'roc_auc': roc_auc, 'g-mean': g-mean, 'f1': f1}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}