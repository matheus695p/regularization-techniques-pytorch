import torch


def multi_acc(y_pred, y_test):
    """
    Calcular la accuracy de un modelo con multiples clases
    Parameters
    ----------
    y_pred : torch tensors
        predicciones.
    y_test : torch tensors
        labels.

    Returns
    -------
    acc : float
        metrica de accuracy en la clasificación.

    """
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc
