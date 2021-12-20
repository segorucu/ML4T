import pandas as pd
import numpy as np
import StrategyLearner as sl
import datetime as dt
from marketsimcode import *
from util import get_data
import matplotlib.pyplot as plt


def author():
    return "sgorucu3"

def calc_stats(fund):
    cumr = fund.iloc[-1]/fund.iloc[0] - 1.
    dr = fund.iloc[1:] / fund.iloc[:-1].values - 1.
    # dr.ix[0] = 0
    adr = dr.mean()
    std = dr.std()
    sharpe = adr / std
    sharpe *= (252 ** 0.5)

    return cumr.iloc[0], adr.iloc[0], std.iloc[0], sharpe.iloc[0]

def main():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.
    symbol = "JPM"
    dict = {"symbol": symbol, "sd": sd, "ed": ed, "sv": sv}

    list_of_impact = [0., 0.01, 0.03, 0.05]
    portvallist = []
    for impact in list_of_impact:
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
        learner.add_evidence(**dict)  # training phase
        df_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)  # testing phase
        orders = pd.DataFrame([[symbol, "BUY", 0]], index=[sd], columns=["Symbol", "Order", "Shares"])
        for i, date in enumerate(df_trades.index):
            val = df_trades.iloc[i][0]
            if val > 0:
                new_order = pd.DataFrame([[symbol, "BUY", abs(val)]], index=[date],
                                         columns=["Symbol", "Order", "Shares"])
                orders = orders.append(new_order)
            elif val < 0:
                new_order = pd.DataFrame([[symbol, "SELL", abs(val)]], index=[date],
                                         columns=["Symbol", "Order", "Shares"])
                orders = orders.append(new_order)
        new_order = pd.DataFrame([[symbol, "BUY", 0]], index=[ed], columns=["Symbol", "Order", "Shares"])
        orders = orders.append(new_order)

        ml_portval = compute_portvals(orders, 100000, commission, impact)
        # print(calc_stats(ml_portval))
        ml_portval /= sv
        portvallist.append(ml_portval)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("(JPM) Impact of Impacts)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    for i, impact in enumerate(list_of_impact):
        plt.plot(portvallist[i], label="impact:" + str(impact))
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('exp2.png')
