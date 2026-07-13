"""Estilo y funciones de plotting reutilizables para los notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PALETTE = "crest"


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"


def plot_global_trend(long_df, ax=None):
    set_style()
    valid = long_df[long_df["is_valid_series"]]
    trend = valid.groupby("fiscal_year_start")["consumption"].sum()
    ax = ax or plt.gca()
    ax.plot(trend.index, trend.values, marker="o", linewidth=2)
    ax.set_title("Consumo doméstico global de café (1990-2020)")
    ax.set_xlabel("Año fiscal (inicio de crop-year)")
    ax.set_ylabel("Consumo (tazas)")
    return ax


def plot_top_countries(long_df, n=15, ax=None):
    set_style()
    valid = long_df[long_df["is_valid_series"]]
    totals = valid.groupby("Country")["consumption"].sum().sort_values(ascending=False).head(n)
    ax = ax or plt.gca()
    sns.barplot(x=totals.values, y=totals.index, ax=ax)
    ax.set_title(f"Top {n} países por consumo doméstico total (1990-2020)")
    ax.set_xlabel("Consumo total (tazas)")
    ax.set_ylabel("")
    return ax


def plot_forecast(years, history, forecast_years, forecast, std=None, ax=None, title=""):
    set_style()
    ax = ax or plt.gca()
    ax.plot(years, history, marker="o", label="Histórico", color="#264653")
    ax.plot(forecast_years, forecast, marker="o", linestyle="--", label="Forecast", color="#e76f51")
    if std is not None:
        forecast = np.asarray(forecast)
        ax.fill_between(forecast_years, forecast - 1.96 * std, forecast + 1.96 * std,
                         color="#e76f51", alpha=0.2, label="IC 95%")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_cluster_scatter(clustered, x="cagr_recent", y="level_mean", cluster_col="cluster_kmeans", ax=None):
    set_style()
    ax = ax or plt.gca()
    sns.scatterplot(data=clustered, x=x, y=y, hue=cluster_col, palette=PALETTE, s=80, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Segmentación de países por patrón de consumo")
    return ax
