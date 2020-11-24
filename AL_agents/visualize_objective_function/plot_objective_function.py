from typing import List, Tuple
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def polyfit_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    poly = PolynomialFeatures(degree=1)
    data = df.to_numpy()
    Y = data[:, -1]  # last column is target
    X = data[:, :-1]
    X_ = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_, Y)

    predicted_acc = model.predict(X_)
    return predicted_acc


def plot_objective_function(
        df: pd.DataFrame,
        axes_: List[str],
        filename_to_save: str = None,
        plot: bool = True
):
    all_betas = ["Uncertainty", "Diversity", "Representative", "Uncertainty_Diversity"]
    df = df.copy(deep=True)
    if True:
        sum = df[all_betas].sum(axis=1).add(1)
        for beta_str in all_betas:
            df[beta_str] = df[beta_str] / sum

    fig = plt.figure()
    x = df[axes_[0]]
    if len(axes_) >= 2:
        y = df[axes_[1]]
    c = df['accuracy']
    if len(axes_) == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.azim = 45
        z = df[axes_[2]]
        img = ax.scatter(x, y, z, c=c, cmap='RdYlGn')
        fig.colorbar(img, ax=ax)
        ax.set_xlabel(axes_[0])
        ax.set_ylabel(axes_[1])
        ax.set_zlabel(axes_[2])

    elif len(axes_) == 2:
        ax = fig.add_subplot()
        img = ax.scatter(x, y, c=c, cmap='RdYlGn')
        ax.set_xlabel(axes_[0])
        ax.set_ylabel(axes_[1])
        fig.colorbar(img)
    elif len(axes_) == 1:
        ax = fig.add_subplot()

        if True:
            # compute univariate polyfit
            coefficients = np.polyfit(x, c, deg=2)
            poly = np.poly1d(coefficients)
            x_min = min(x.values)
            x_max = max(x.values)
            new_x = np.linspace(x_min, x_max)
            new_c = poly(new_x)
        else:
            df_ = df.sort_values(by=axes_)
            new_x = df_[axes_]
            new_c = polyfit_df(df)
        plt.scatter(x, c)
        plt.plot(new_x, new_c)

        ax.set_xlabel(axes_[0])
        ax.set_ylabel('accuracy')

    # plt.title(f"Accuracy dependent on relative weight(s) of heuristics(s)")
    plt.tight_layout()

    if filename_to_save is not None:
        directory = os.path.dirname(filename_to_save)
        Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(filename_to_save)
    if plot:
        plt.show()
