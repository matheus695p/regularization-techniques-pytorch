import numpy as np
import pandas as pd
from src.preprocessing_module import (exponential_features,
                                      polinomial_features,
                                      log_features, selection_by_correlation,
                                      targets_galaxies_simple_encode)
# from src.visualizations import pairplot_sns

# leer datos
data = pd.read_csv("data/galaxies.csv")

data = data[['NVOTE', 'P_EL', 'P_CW', 'P_ACW', 'P_EDGE', 'P_DK', 'P_MG',
             'P_CS', 'P_EL_DEBIASED', 'P_CS_DEBIASED', 'SPIRAL',
             'ELLIPTICAL', 'UNCERTAIN']]

for col in data.columns:
    data.rename(columns={col: col.lower()}, inplace=True)

# columnas a hacer feature engeniering
columns = list(data.columns)

# columnas de predicción
targets = ['spiral', 'elliptical', 'uncertain']
for col in targets:
    columns.remove(col)

# visualizar un poco como se van a dar las corr
# features
# pairplot_sns(data, columns, name="Viz")
# # targets
# pairplot_sns(data, targets, name="Viz")

# exponencial
data = exponential_features(data, columns)
# loagaritmo
data = log_features(data, columns)
# raiz
data = polinomial_features(data, columns, grade=0.5)
# potencia 2
data = polinomial_features(data, columns, grade=2)
# potencia 3
data = polinomial_features(data, columns, grade=3)
# potencia 5
data = polinomial_features(data, columns, grade=5)

# seleccionar por correlación
corr = selection_by_correlation(data, threshold=0.9)
corr.replace([np.inf, -np.inf], np.nan, inplace=True)

# droping columns
nans = pd.DataFrame(corr.isna().sum(), columns=["counter"])
nans.reset_index(drop=False, inplace=True)
nans.rename(columns={"index": "column"}, inplace=True)
nans["percentage"] = nans["counter"] / len(corr) * 100

# no dejar columnas con nans
droping_cols = nans[nans["percentage"] > 0]["column"].to_list()
corr.drop(columns=droping_cols, inplace=True)

# dataframe de salida con los targets en simple encode
output = targets_galaxies_simple_encode(corr, targets)

# guardar dataframe
output.to_csv("data/glaxies_featured.csv", index=False)
