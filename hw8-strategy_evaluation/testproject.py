import ManualStrategy as ms
import StrategyLearner as sl
import experiment1
import experiment2
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from marketsimcode import *
from util import get_data


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


if __name__ == "__main__":

    #   Manual in-sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    impact = 0.005
    commission = 9.95
    symbol = "JPM"
    dict = {"symbol": symbol, "sd": sd, "ed": ed, "sv": sv}
    manual_orders = ms.testPolicy(**dict)
    manual_portval = compute_portvals(manual_orders, sv, commission, impact)
    manual_portval /= sv
    buy = manual_orders[manual_orders["Order"] == "BUY"]
    buydate = buy.index
    sell = manual_orders[manual_orders["Order"] == "SELL"]
    selldate = sell.index

    columns = manual_orders.columns
    pv = get_data([symbol], pd.date_range(sd, ed)).index
    bench_orders = pd.DataFrame([[symbol, "BUY", 1000], [symbol, "SELL", 1000]], index=[pv[0], pv[-1]], columns=columns)
    bench_portval = compute_portvals(bench_orders, sv, commission, impact)
    bench_portval /= sv

    ymin = min(bench_portval.min()[0], manual_portval.min()[0])
    ymax = max(bench_portval.max()[0], manual_portval.max()[0])

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("(JPM) In-Sample Benchmark vs Manual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    plt.plot(bench_portval, 'green', label="Benchline")
    plt.plot(manual_portval, 'red', label="Manual")
    plt.vlines(buydate, ymin=ymin, ymax=ymax, label="LONG", colors='blue')
    plt.vlines(selldate, ymin=ymin, ymax=ymax, label="SHORT", colors='black')
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('manual_in.png')


    # stats= calc_stats(bench_portval)
    # print(stats)
    # stats= calc_stats(manual_portval)
    # print(stats)

    #   Manual out-sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    impact = 0.005
    commission = 9.95
    symbol = "JPM"
    dict = {"symbol": symbol, "sd": sd, "ed": ed, "sv": sv}
    manual_orders = ms.testPolicy(**dict)
    manual_portval = compute_portvals(manual_orders, sv, commission, impact)
    manual_portval /= sv
    buy = manual_orders[manual_orders["Order"] == "BUY"]
    buydate = buy.index
    sell = manual_orders[manual_orders["Order"] == "SELL"]
    selldate = sell.index

    columns = manual_orders.columns
    pv = get_data([symbol], pd.date_range(sd, ed)).index
    bench_orders = pd.DataFrame([[symbol, "BUY", 1000], [symbol, "SELL", 1000]], index=[pv[0], pv[-1]], columns=columns)
    bench_portval = compute_portvals(bench_orders, sv, commission, impact)
    bench_portval /= sv

    ymin = min(bench_portval.min()[0], manual_portval.min()[0])
    ymax = max(bench_portval.max()[0], manual_portval.max()[0])

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("(JPM) Out-Sample Benchmark vs Manual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    plt.plot(bench_portval, 'green', label="Benchline")
    plt.plot(manual_portval, 'red', label="Manual")
    plt.vlines(buydate, ymin=ymin, ymax=ymax, label="LONG", colors='blue')
    plt.vlines(selldate, ymin=ymin, ymax=ymax, label="SHORT", colors='black')
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('manual_out.png')
    # stats = calc_stats(bench_portval)
    # print(stats)
    # stats = calc_stats(manual_portval)
    # print(stats)

    #   Manual out-sample 2
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    impact = 0.0005
    commission = 9.95
    symbol = "JPM"
    dict = {"symbol": symbol, "sd": sd, "ed": ed, "sv": sv}
    manual_orders = ms.testPolicy(**dict)
    manual_portval = compute_portvals(manual_orders, sv, commission, impact)
    manual_portval /= sv
    buy = manual_orders[manual_orders["Order"] == "BUY"]
    buydate = buy.index
    sell = manual_orders[manual_orders["Order"] == "SELL"]
    selldate = sell.index

    columns = manual_orders.columns
    pv = get_data([symbol], pd.date_range(sd, ed)).index
    bench_orders = pd.DataFrame([[symbol, "BUY", 1000], [symbol, "SELL", 1000]], index=[pv[0], pv[-1]], columns=columns)
    bench_portval = compute_portvals(bench_orders, sv, commission, impact)
    bench_portval /= sv

    ymin = min(bench_portval.min()[0], manual_portval.min()[0])
    ymax = max(bench_portval.max()[0], manual_portval.max()[0])

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("(JPM) Out-Sample Benchmark vs Manual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")
    plt.plot(bench_portval, 'green', label="Benchline")
    plt.plot(manual_portval, 'red', label="Manual")
    plt.vlines(buydate, ymin=ymin, ymax=ymax, label="LONG", colors='blue')
    plt.vlines(selldate, ymin=ymin, ymax=ymax, label="SHORT", colors='black')
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('manual_out2.png')




    experiment1.main()
    experiment2.main()





