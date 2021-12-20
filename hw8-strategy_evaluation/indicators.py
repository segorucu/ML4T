import datetime as dt
import pandas as pd
import numpy as np
from util import get_data, plot_data
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def author():
    return "sgorucu3"


def indicator(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)):
    adjclose = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY")
    momentum(adjclose)
    bollinger(adjclose)
    macd(adjclose)
    stochastic_oscillator(symbol, sd, ed)
    aroon_oscillator(adjclose)

def aroon_oscillator(df):
    Aroon = pd.DataFrame(np.zeros((len(df),3)),index=df.index, columns = ["Up", "Down", "Oscillator"])
    for count, date in enumerate(df.index):
        if count < 25:
            Aroon.loc[date] = np.nan
            continue
        a = df.iloc[count-25:count]
        maxdate = a.idxmax()[0]
        mindate = a.idxmin()[0]
        argmax = a.index.get_loc(maxdate)
        argmin = a.index.get_loc(mindate)
        Aroon.loc[date]["Up"] = 100 * argmax / 25
        Aroon.loc[date]["Down"] = 100 * argmin / 25
        Aroon.loc[date]["Oscillator"] = Aroon.loc[date]["Up"] - Aroon.loc[date]["Down"]

    # fig, (ax, ax2) = plt.subplots(2, 1)
    # ax.set_title("Aroon Up, Aroon down")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Percentage")
    # ax.plot(Aroon["Up"], 'g', label="Aroon Up")
    # ax.plot(Aroon["Down"], 'b', label="Aroon Down")
    # ax.legend(loc='center', bbox_to_anchor=(0.13, 1.2))
    # ax2.plot(Aroon["Oscillator"], 'purple', label="Aroon Oscillator")
    # ax2.plot(df, label="prices")
    # ax2.set_xlabel("Date")
    # ax2.set_ylabel("Percentage")
    # ax2.set_title("Aroon Oscillator")
    # ax2.axhline(y=0., color='r', linestyle=":")
    # # plt.xticks(rotation=60)
    # plt.tight_layout()
    # plt.savefig('Aroon.png')


    return Aroon["Oscillator"]

def stochastic_oscillator(symbol, sd, ed):
    close = get_data([symbol], pd.date_range(sd, ed), colname="Close").drop(columns="SPY")
    high = get_data([symbol], pd.date_range(sd, ed), colname="High").drop(columns="SPY")
    low = get_data([symbol], pd.date_range(sd, ed), colname="Low").drop(columns="SPY")
    # symbol = df.columns[0]
    highest = high.rolling(14).max()
    lowest = low.rolling(14).min()
    K = (close - lowest) * 100 / (highest - lowest)
    D = K.rolling(3).mean()

    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.gca()
    # ax.set_title("Stochastic Oscillator")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Percentage")
    # plt.plot(K, 'g', label="%K")
    # plt.plot(D, 'b', label="%D")
    # ax.axhline(y=20, color='r', linestyle=":")
    # ax.axhline(y=80, color='r', linestyle=":")
    # ax.legend(loc='center', bbox_to_anchor=(0.13, 1.2))
    # # plt.xticks(rotation=80)
    # plt.tight_layout()
    # plt.savefig('stochastic_osc.png')

    return K, D


def bollinger(df):
    df = df / df.iloc[0]
    mean_df = df.rolling(20, min_periods=1).mean()
    std_df = df.rolling(20, min_periods=1).std()
    bolu = mean_df + 2 * std_df
    bolp = mean_df - 2 * std_df
    bbp = (df - mean_df) / (2 * std_df)
    # fig, (ax, ax2) = plt.subplots(2, 1)
    # # fig.figure(figsize=(6, 4))
    # # ax = fig.gca()
    # ax.set_title("Bollinger Bands (20)")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # # ax.plot(mean_df, 'g', label="20 days SMA")
    # ax.plot(df, 'b', label='Normalized Stock AAPL')
    # ax.plot(bolu, 'r')
    # ax.plot(bolp, 'r')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.22, 0.4))
    # ax2.plot(bbp)
    # ax2.axhline(y=1., color='r', linestyle=":")
    # ax2.axhline(y=-1., color='r', linestyle=":")
    # ax.set_xlabel("Date")
    # ax2.set_ylabel("Bollinger Bands Percentage")
    # # plt.xticks(rotation=80)
    #
    # plt.tight_layout()
    # plt.savefig('bollinger.png')



    return bbp


def sma(df):
    df = df / df.iloc[0]
    mean_short = df.rolling(50, min_periods=1).mean()
    mean_long = df.rolling(200, min_periods=1).mean()
    sma_cross = mean_short / mean_long
    # fig, (ax, ax2) = plt.subplots(2, 1)
    # ax.set_title("SMA(20)")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # ax.plot(mean_short, 'g', label="SMA(50)")
    # ax.plot(mean_long, 'r', label="SMA(200)")
    # ax.plot(df, 'b', label='Normalized Stock JPM')
    # # ax.plot(bolu, 'r')
    # # ax.plot(bolp, 'r')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.22, 0.4))
    # ax2.plot(sma_cross)
    # ax2.axhline(y=1., color='r', linestyle=":")
    # ax.set_xlabel("Date")
    # ax2.set_ylabel("Price / SMA")
    # # plt.xticks(rotation=80)
    #
    # plt.tight_layout()
    # plt.savefig('sma.png')
    return sma_cross


def momentum(df):
    df = df / df.iloc[0]
    momentum_df = df / df.shift(20) - 1
    # momentum_df = momentum_df.dropna()

    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.gca()
    # ax.set_title("Price Momentum")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # plt.plot(momentum_df, 'g', label="Momentum(20)")
    # plt.plot(df, 'b', label='Normalized Stock JPM')
    # plt.axhline(y=0., color='r', linestyle=':')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.22, 0.7))
    # plt.xticks(rotation=80)
    # plt.tight_layout()
    # plt.savefig('momentum.png')
    return momentum_df


def macd(df):
    df = df / df.iloc[0]
    symbol = df.columns

    # df["sma"] = df[symbol].rolling(20).mean()
    df["ema12"] = df[symbol].ewm(span=12, adjust=False).mean()
    df["ema26"] = df[symbol].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # fig = plt.figure(figsize=(6, 4))
    # ax = fig.gca()
    # ax.set_title("MACD")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # plt.plot(df["macd"], 'g', label="MACD")
    # plt.plot(df["signal"], 'r', label="Signal")
    # ax.legend(loc=4)
    # plt.xticks(rotation=80)
    # plt.tight_layout()
    # plt.savefig('macd.png')

    ind = df["macd"] / df["signal"]
    # ind[ind<0.9] = 0.9
    # ind[ind>1.1] = 1.1
    return ind
