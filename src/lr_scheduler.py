import torch


class LRScheduler():
    """
    Scheduler de tasas de aprendizaje. Si la pérdida de validación no
    disminuye para un dado número de épocas de "paciencia", la tasa de
    aprendizaje disminuirá en por el "factor" dado.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        Constructor del scheduler

        Parameters
        ----------
        optimizer : torch.optimizer
            optimizador que se usa.
        patience : int, optional
            épocas de paciencia. The default is 5.
        min_lr : float, optional
            minima tasa de aprendizaje. The default is 1e-6.
        factor : float, optional
            número entre 0-1 para disminuir la tasa de aprendizaje.
            The default is 0.5.

        Returns
        -------
        LR Scheduler.

        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
