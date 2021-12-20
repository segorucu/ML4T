### Manual Strategy
import numpy as np
import pandas as pd
import datetime as dt
from util import get_data, plot_data
from copy import deepcopy
from indicators import *


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

    start_date = sd
    end_date = ed
    symbols = [symbol]
    portvals = get_data(symbols, pd.date_range(start_date, end_date))
    dfprices = portvals[symbols]  # remove SPY
    dateindex = portvals.index
    symbols = np.append(symbols, "CASH")
    zeros = np.zeros((dateindex.shape[0], len(symbols)))
    dftrades = pd.DataFrame(zeros, index=dateindex, columns=symbols)
    dfholdings = deepcopy(dftrades)
    dfholdings.iloc[0]["CASH"] = sv

    Aroon = aroon_oscillator(dfprices)
    # K, D = stochastic_oscillator(symbol, sd, ed)
    bbp= bollinger(dfprices)
    # smar = sma(dfprices)
    # mom = momentum(dfprices)
    macdrat = macd(dfprices)

    # fig, ax1 = plt.subplots()
    # ax1.plot(list(range(100)), dfprices.iloc[0:100])
    # ax2 = ax1.twinx()
    # ax2.plot(list(range(100)), macdrat.iloc[0:100], color = 'blue')
    # # ax2.set_ylim([0.1, 10])
    # plt.savefig('macd2.png')

    orders = pd.DataFrame([[symbol, "BUY", 1000]], index=dateindex[0:1], columns=["Symbol","Order","Shares"])
    lastorderdate = dateindex[0]
    for i, date in enumerate(dateindex):
        if i == len(dateindex)-1:
            if orders.loc[lastorderdate]["Order"] == "BUY":
                decision = "SELL"
            else:
                decision = "BUY"
            new_order = pd.DataFrame([[symbol, decision, 1000]], index=[date], columns=["Symbol", "Order", "Shares"])
            orders = orders.append(new_order)
            break
        if i < 30:
            continue
        sell = 0
        if macdrat.iloc[i] < 1. and macdrat.iloc[i-1] > 1.:
            sell += 1
        if bbp.iloc[i][0] > 1.:
            sell += 1
        # if smar.iloc[i][0] < 1.:
        #     sell += 1
        if Aroon.iloc[i] < 0. and Aroon.iloc[i-1] > 0.:
            sell += 1
        # if mom.iloc[i][0] < 0. and mom.iloc[i-1][0] > 0.:
        #     sell += 1
        decision = "BUY"
        if sell >= 1:
            decision = "SELL"
        if decision == "BUY" and orders.shape[0] > 1:
            if orders.loc[lastorderdate]["Order"] == "SELL":
                if dfprices.loc[lastorderdate][0] < dfprices.loc[date][0]:
                    if date - lastorderdate < dt.timedelta(days=2):
                        decision = "SELL"
        if decision == orders.loc[lastorderdate]["Order"]:
            continue
        shares = 2000
        new_order = pd.DataFrame([[symbol, decision, shares]], index=[date], columns=["Symbol","Order","Shares"])
        orders = orders.append(new_order)
        lastorderdate = date

    return orders

def author():
    return "sgorucu3"