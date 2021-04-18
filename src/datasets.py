import torch
from torch.utils.data import Dataset


class TrainDataset:
    """
    Divide targets y features recibiendo un array de los conjuntos de datos
    """

    def __init__(self, features, targets):
        """
        Constructor
        Parameters
        ----------
        features : array
            features.
        targets : array
            targets.

        Returns
        -------
        Diccionario para darselo al data_loader.

        """
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class TestDataset:
    """
    Transforma a tensores los features recibiendo un array de los conjuntos
    de datos de test
    """

    def __init__(self, features):
        """
        Constructor
        Parameters
        ----------
        features : array
            features.
        Returns
        -------
        Diccionario para darselo al data_loader.

        """
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct


class ClassifierDataset(Dataset):
    """
    Divide targets y features recibiendo un array de los conjuntos de datos
    por indice
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)
