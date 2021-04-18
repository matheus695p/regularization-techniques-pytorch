import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.preprocessing_module import (get_class_distribution,
                                      get_galaxies_distribution)
sns.set(font_scale=1.5)
plt.style.use('dark_background')


def pairplot_sns(df, columns, name="Entrenamiento"):
    """
    Pairplot del dataframe insertado
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    Returns
    -------
    Pairplot.
    """
    df = df[columns]
    length = len(list(df.columns))

    sns.set(font_scale=0.75 * length)
    plt.style.use('dark_background')
    pplot = sns.pairplot(df)
    pplot.fig.set_figwidth(7.5*length)
    pplot.fig.set_figheight(7.5*length)
    pplot.set(title=name)

    # counter = 0
    # while counter < length:
    #     counter += 2
    #     print(counter)
    #     alpha = True
    #     try:
    #         cols = columns[counter:counter+2]
    #     except Exception:
    #         cols = columns[counter:]
    #         if len(cols) == 0:
    #             alpha = False
    #     if alpha:
    #         print("columnas", cols)
    #         sns.set(font_scale=0.75 * length)
    #         plt.style.use('dark_background')
    #         pplot = sns.pairplot(df[cols])
    #         pplot.fig.set_figwidth(7.5*length)
    #         pplot.fig.set_figheight(7.5*length)
    #         pplot.set(title=name)


def pairgrid_plot(df, name="Entrenamiento"):
    """
    Pair grid del dataframe insertado
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    Returns
    -------
    Pairgrid plot.
    """
    length = len(list(df.columns))
    if length <= 2:
        sns.set(font_scale=0.75 * length)
        plt.style.use('dark_background')
        g = sns.PairGrid(df)
        g.map_upper(sns.histplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        g.fig.set_figwidth(7.5*length)
        g.fig.set_figheight(7.5*length)
        g.set(title=name)
    else:
        sns.set(font_scale=15)
        plt.style.use('dark_background')
        g = sns.PairGrid(df)
        g.map_upper(sns.histplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        g.fig.set_figwidth(length/2)
        g.fig.set_figheight(length/2)
        g.set(title=name)


def violinplot(df, col, name="Entrenamiento"):
    """
    Violin plot de la columna de un dataframe
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    Returns
    -------
    Violin plot de la columna.
    """
    rcParams['figure.figsize'] = 15, 15
    g = sns.violinplot(y=col, data=df)
    g.set_yticklabels(g.get_yticks(), size=30)
    g.axes.set_title(name, fontsize=40)
    g.set_ylabel(col, fontsize=40)
    plt.show()


def kernel_density_estimation(df, col, name="Entrenamiento", bw_adjust=0.1):
    """
    Estimación de la densidad a través de un kernel
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    bw_adjust : float, optional
        Ajuste de la distribución. The default is 0.1.
    Returns
    -------
    Estimación de la distribución de la columna.
    """
    sns.set(font_scale=1.5)
    plt.style.use('dark_background')
    pplot = sns.displot(df, x=col, kind="kde", bw_adjust=bw_adjust)
    pplot.fig.set_figwidth(10)
    pplot.fig.set_figheight(8)
    pplot.set(title=name)


def watch_distributiions(y_train, y_val, y_test):
    """
    Ver las distribuciones de los targets
    Parameters
    ----------
    y_train : numpy array
        DESCRIPTION.
    y_val : numpy array
        DESCRIPTION.
    y_test : numpy array
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    # entrenamiento
    sns.barplot(data=pd.DataFrame.from_dict([
        get_class_distribution(y_train)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[0]).set_title('Distribución en Train Set')
    # validación
    sns.barplot(data=pd.DataFrame.from_dict([
        get_class_distribution(y_val)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[1]).set_title('Distribución en Val Set')
    # testing
    sns.barplot(data=pd.DataFrame.from_dict([
        get_class_distribution(y_test)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[2]).set_title('Distribución en Test Set')


def watch_galaxies_distributiions(y_train, y_val, y_test):
    """
    Ver las distribuciones de los targets
    Parameters
    ----------
    y_train : numpy array
        DESCRIPTION.
    y_val : numpy array
        DESCRIPTION.
    y_test : numpy array
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    # entrenamiento
    sns.barplot(data=pd.DataFrame.from_dict([
        get_galaxies_distribution(y_train)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[0]).set_title('Distribución en Train Set')
    # validación
    sns.barplot(data=pd.DataFrame.from_dict([
        get_galaxies_distribution(y_val)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[1]).set_title('Distribución en Val Set')
    # testing
    sns.barplot(data=pd.DataFrame.from_dict([
        get_galaxies_distribution(y_test)]).melt(),
        x="variable", y="value", hue="variable",
        ax=axes[2]).set_title('Distribución en Test Set')


def torch_classification_visualizer(loss_stats, accuracy_stats):
    """
    Crear dataframes

    Parameters
    ----------
    loss_stats : list
        decrecimiento del loss, sacado del entrenamiento [val loss].
    accuracy_stats : list
        decrecimiento de accuracy sacado del entrenamiento [val loss].

    Returns
    -------
    plots.

    """
    # Crear dataframes
    train_val_acc_df = pd.DataFrame.from_dict(
        accuracy_stats).reset_index().melt(
            id_vars=['index']).rename(columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(
        id_vars=['index']).rename(columns={"index": "epochs"})
    # Plotear dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value",
                 hue="variable",  ax=axes[0]).set_title(
                     'Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value",
                 hue="variable", ax=axes[1]).set_title(
                     'Train-Val Loss/Epoch')


def plot_confusion_matrix(df_confusion, title='Matriz de confusion',
                          cmap=plt.cm.hot):
    """
    Visualizar la matriz de confusión de la clasificación

    Parameters
    ----------
    df_confusion : dataframe
        matriz de confusión.
    title : string, optional
        titulo del gráficoo. The default is 'Matriz de confusion'.
    cmap : TYPE, optional
        DESCRIPTION. The default is plt.cm.gray_r.

    Returns
    -------
    Plot de la matriz de confusión.

    """
    plt.matshow(df_confusion, cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
