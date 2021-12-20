import ManualStrategy as ms
import pandas as pd
import numpy as np
import datetime as dt
from marketsimcode import *
from util import get_data
import matplotlib.pyplot as plt
import StrategyLearner as sl


def author():
    return "sgorucu3"


def main():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    impact = 0.005
    commission = 9.95
    symbol = "JPM"
    dict = {"symbol": symbol, "sd": sd, "ed": ed, "sv": sv}
    manual_orders = ms.testPolicy(**dict)
    manual_portval = compute_portvals(manual_orders, 100000, commission, impact)
    manual_portval /= sv

    columns = manual_orders.columns
    pv = get_data([symbol], pd.date_range(sd, ed)).index
    bench_orders = pd.DataFrame([[symbol, "BUY", 1000], [symbol, "SELL", 1000]], index=[pv[0], pv[-1]], columns=columns)
    bench_portval = compute_portvals(bench_orders, 100000, commission, impact)
    bench_portval /= sv

    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    learner.add_evidence(**dict)  # training phase
    # df_trades = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
    #                                sv=sv)  # testing phase
    df_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)  # testing phase
    orders = pd.DataFrame([[symbol, "BUY", 0]], index=[sd], columns=["Symbol","Order","Shares"])
    for i, date in enumerate(df_trades.index):
        val = df_trades.iloc[i][0]
        if val > 0:
            new_order = pd.DataFrame([[symbol, "BUY", abs(val)]], index=[date], columns=["Symbol", "Order", "Shares"])
            orders = orders.append(new_order)
        elif val < 0:
            new_order = pd.DataFrame([[symbol, "SELL", abs(val)]], index=[date], columns=["Symbol", "Order", "Shares"])
            orders = orders.append(new_order)
    new_order = pd.DataFrame([[symbol, "BUY", 0]], index=[ed], columns=["Symbol", "Order", "Shares"])
    orders = orders.append(new_order)

    ml_portval = compute_portvals(orders, 100000, commission, impact)
    ml_portval /= sv

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("(JPM) Exp1: Manual vs Benchline vs Random Forest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    plt.plot(bench_portval, 'g', label="Benchline")
    plt.plot(manual_portval, 'r', label="Manual")
    plt.plot(ml_portval, 'b', label="Random Forest Tree")
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('exp1.png')


