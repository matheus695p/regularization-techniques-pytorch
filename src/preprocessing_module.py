import numpy as np
import pandas as pd


def exponential_features(df, columns):
    """
    Agregar polinomial features al dataframe en las columnas mencionadas
    Parameters
    ----------
    df : dataframe
        variables input.
    columns : list
        columnas a aplicar.

    Returns
    -------
    df : dataframe
        dataframe con los nuevos features.

    """
    for col in columns:
        new_col = str(col) + "_exponential"
        df[new_col] = df[col].apply(lambda x: exponential_compute(x))
    return df


def polinomial_features(df, columns, grade=2):
    """
    Agregar polinomial features al dataframe en las columnas mencionadas
    Parameters
    ----------
    df : dataframe
        variables input.
    columns : list
        columnas a aplicar.
    grade : int, optional
        grado del polinomio. The default is 2.

    Returns
    -------
    df : dataframe
        dataframe con los nuevos features.

    """
    for col in columns:
        new_col = str(col) + f"_poly_{str(grade)}"
        df[new_col] = df[col].apply(lambda x: polinomial_compute(x, grade))
    return df


def exponential_compute(x):
    """
    Retorna el vector elevado a la potencia grade

    Parameters
    ----------
    x : numpy array
        numero.

    Returns
    -------
    float
        exponencial del número.

    """
    return np.exp(x)


def polinomial_compute(x, grade=2):
    """
    Retorna el vector elevado a la potencia grade

    Parameters
    ----------
    x : numpy array
        numero.
    grade : int, optional
        grado de la potencia. The default is 2.

    Returns
    -------
    float
        potencia del número.

    """
    return np.power(x, grade)


def add_lagged_variables(df, columns, nr_of_lags=1):
    """
    Agregar variables pasadas del dataframe con el que se esta trabajando
    Parameters
    ----------
    df_input : df
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas.
    """
    for i in range(1, nr_of_lags+1):
        for col in columns:
            lagged_column = col + f'_lagged_{i}'
            df[lagged_column] = df[col].shift(i)
    return df


def log_features(df, columns):
    """
    Agregar logaritmo de las variables
    Parameters
    ----------
    df : dataframe
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas con logaritmo.
    """
    for col in columns:
        log_column = 'log_' + col
        df[log_column] = df[col].apply(lambda x: np.log(x))
    return df


def log_lagged_variables(df, columns, nr_of_lags=1):
    """
    Agregar variables aplicando logaritmo en ellas
    Parameters
    ----------
    df : dataframe
        dataframe a operar.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas con logaritmo.
    """
    for i in range(1, nr_of_lags+1):
        for col in columns:
            lagged_column = col + f'_log_lagged_{i}'
            df[lagged_column] = df[col].shift(i)
            df[lagged_column] = df[lagged_column].apply(lambda x: np.log(x))
    return df


def convert_df_float(df):
    """
    Pasa por las columnas tratando de convertirlas a float64
    Parameters
    ----------
    df : dataframe
        df de trabajo.
    Returns
    -------
    df : dataframe
        df con las columnas númericas en float.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception as e:
            print(e)
    df.reset_index(drop=True, inplace=True)
    return df


def log_transform(number):
    """
    Aplicar logaritmo en las variables
    Parameters
    ----------
    number : float
        aplicar logaritmos.
    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    return np.log(number)


def downcast_dtypes(df):
    """
    Función super util para bajar la cantidad de operaciones flotante que
    se van a realizar en el proceso de entrenamiento de la red
    Parameters
    ----------
    df : dataframe
        df a disminuir la cantidad de operaciones flotantes.
    Returns
    -------
    df : dataframe
        dataframe con los tipos int16 y float32 como formato número
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def lowwer_rename(df):
    """
    Renombrar nombres de las columnas
    Parameters
    ----------
    df : dataframe
        dataframe.
    Returns
    -------
    df : dataframe
        dataframe con los nombres en minusculas.
    """
    for col in df.columns:
        new_col = col.lower()
        print("Renombrando columna:", col, "-->", new_col)
        df.rename(columns={col: new_col}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def selection_by_correlation(dataset, threshold=0.5):
    """
    Selecciona solo una de las columnas altamente correlacionadas y elimina
    la otra
    Parameters
    ----------
    dataset : dataframe
        dataset sin la variable objectivo.
    threshold : float
        modulo del valor threshold de correlación pearson.
    Returns
    -------
    dataset : dataframe
        dataset con las columnas eliminadas.
    """
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            cond1 = (corr_matrix.iloc[i, j] >= threshold)
            if cond1 and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]
    dataset = dataset.reset_index(drop=True)
    return dataset


def nn_preparation(df, target_col):
    """
    Hacer la preparación de la red neuronal
    Parameters
    ----------
    df : datarame
        dataframe con todas las variables.
    target_col : string or list
        nombre de la/as columna/as target/s.
    Returns
    -------
    x : array
        x en numpy.
    y : array
        target en numpy.
    """
    if len(target_col) == 1:
        y = df[[target_col]]
    else:
        y = df[target_col]
    x = df.drop(columns=target_col)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


def numpy_array_indexes(array):
    """
    Retorna indices de un array

    Parameters
    ----------
    array : numpy array
        array del cual se quiere saber indices.

    Returns
    -------
    index
        indices del array.

    """
    df = pd.DataFrame(array)
    return df.index


def get_class_distribution(obj):
    """
    Visualizar la distribución de las clases en la clasificación

    Parameters
    ----------
    obj : obj
        targets.

    Returns
    -------
    count_dict : dict
        diccionario para ver las distribuciones.

    """
    # count_dict = {
    #     "spiral": 0,
    #     "elliptical": 0,
    #     "uncertain": 0,
    # }

    # for i in obj:
    #     if i == 0:
    #         count_dict['spiral'] += 1
    #     elif i == 1:
    #         count_dict['elliptical'] += 1
    #     elif i == 2:
    #         count_dict['uncertain'] += 1
    #     else:
    #         print("Check classes.")
    # return count_dict

    count_dict = {
        "rating_3": 0,
        "rating_4": 0,
        "rating_5": 0,
        "rating_6": 0,
        "rating_7": 0,
        "rating_8": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['rating_3'] += 1
        elif i == 1:
            count_dict['rating_4'] += 1
        elif i == 2:
            count_dict['rating_5'] += 1
        elif i == 3:
            count_dict['rating_6'] += 1
        elif i == 4:
            count_dict['rating_7'] += 1
        elif i == 5:
            count_dict['rating_8'] += 1
        else:
            print("Check classes.")

    return count_dict


def get_galaxies_distribution(obj):
    """
    Visualizar la distribución de las clases en la clasificación

    Parameters
    ----------
    obj : obj
        targets.

    Returns
    -------
    count_dict : dict
        diccionario para ver las distribuciones.

    """
    count_dict = {
        "spiral": 0,
        "elliptical": 0,
        "uncertain": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['spiral'] += 1
        elif i == 1:
            count_dict['elliptical'] += 1
        elif i == 2:
            count_dict['uncertain'] += 1
        else:
            print("Check classes.")
    return count_dict


def replace_ones_by_number(number, number2replace):
    """
    Donde haya un número 1 reemplazarlo por number2replace

    Parameters
    ----------
    number : int
        numero en el que se pasa el apply lambda.
    number2replace : int
        numero que se quiere reemplazar.

    Returns
    -------
    number : int
        numero de la clase.

    """
    if number == 1:
        number = number2replace
    return number


def get_names(col):
    """
    Poner en string los resultados
    Parameters
    ----------
    col : int
        int class.

    Returns
    -------
    name : string
        nombre de la galaxias.

    """
    if col == 0:
        name = "spiral"
    if col == 1:
        name = "elliptical"
    if col == 2:
        name = "uncertain"
    return name


def targets_galaxies_simple_encode(df, targets):
    """
    Preprocessing para tener todo en un simple encode de cada uno de las
    galaxias en el dataset de galaxias

    Parameters
    ----------
    df : dataframe
        dataframe de entrenamiento.
    targets : list
        columnas que son targets y que están como one-hot-encoder.

    Returns
    -------
    df : datframe
        dataframe con la columna class y en simple encode.

    """
    for col in targets:
        if col == "spiral":
            number2replace = 0
        if col == "elliptical":
            number2replace = 1
        if col == "uncertain":
            number2replace = 2
        print("Reemplazando", col, "por ", number2replace)

        df[col] = df[col].apply(
            lambda x: replace_ones_by_number(x, number2replace))
    df["class"] = df[targets].sum(axis=1)
    df["class"] = df["class"].apply(lambda x: get_names(x))
    df.drop(columns=targets, inplace=True)
    return df
