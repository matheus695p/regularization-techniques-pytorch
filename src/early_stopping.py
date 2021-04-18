import torch
import numpy as np


class EarlyStopping():
    """
    Early stopping para detener el entrenamiento cuando el loss de validación
    no mejora después cierto número de épocas.
    """

    def __init__(self, patience=7, verbose=True, delta=0,
                 path='checkpoint.pt', trace_func=print):
        """

        Parameters
        ----------
        patience : int, optional
            Cuánto tiempo esperar después de que se mejoró la última pérdida
            de validación. The default is 7.
        verbose : bolean, optional
            Si es verdadero, imprime un mensaje para cada mejora de pérdida
            de validación. The default is False.
        delta : float, optional
            Cambio mínimo en la cantidad monitoreada para calificar
            como una mejora. The default is 0.
        path : TYPE, optional
            Ruta para guardar el punto de control. The default is
            'checkpoint.pt'.
        trace_func : TYPE, optional
            impresión del seguimiento. The default is print.

        Returns
        -------
        None.

        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping: {self.counter} sobre {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Error de validación bajo ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Guardando modelo ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
