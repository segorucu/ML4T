import datetime as dt
import pandas as pd
import numpy as np
from util import get_data, plot_data
from copy import deepcopy
import marketsimcode
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
    return "sgorucu3"


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    adjclose = get_data([symbol], pd.date_range(sd, ed)).drop(columns="SPY")
    dateindex = adjclose.index
    orders = pd.DataFrame()
    holding = 0
    for count, date in enumerate(dateindex):
        if date == dateindex[-1]:
            break
        if adjclose.iloc[count+1][symbol] > adjclose.loc[date][symbol]:
            order = 1000 - holding
            holding += order
        elif adjclose.iloc[count+1][symbol] < adjclose.loc[date][symbol]:
            order = -1000 - holding
            holding += order
        df = pd.Series(order,index=[date])
        if len(orders) > 0:
            orders = orders.append(df)
        else:
            orders = deepcopy(df)

    orders_new = deepcopy(orders)
    orders_new = orders_new[orders_new.values != 0]
    orders_new = pd.DataFrame(orders_new.values, index=orders_new.index, columns=["decision"])
    orders_new["Order"] = np.where(orders_new.values > 0, "BUY", "SELL")
    orders_new["Symbol"] = symbol
    orders_new["Shares"] = abs(orders_new["decision"])
    orders_new = orders_new.drop(columns="decision")
    orders_new = orders_new[["Symbol", "Order", "Shares"]]
    optimal_portfolio = marketsimcode.compute_portvals(orders_new,start_val=sv,commission=0,impact=0)

    benchmark_orders = pd.DataFrame(index=[optimal_portfolio.index[0],optimal_portfolio.index[-1]],columns=["Symbol", "Order", "Shares"])
    benchmark_orders["Symbol"] = symbol
    benchmark_orders.iloc[0]["Shares"] = 1000
    benchmark_orders.iloc[-1]["Shares"] = 0
    benchmark_orders["Order"] = "BUY"
    benchmark = marketsimcode.compute_portvals(benchmark_orders, start_val=sv, commission=0, impact=0)

    norm_optimal_portfolio = optimal_portfolio / optimal_portfolio.iloc[0]  #red
    norm_benchmark = benchmark / benchmark.iloc[0]  #green

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("Normalized Benchmark vs Optimal Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value ($)")
    plt.plot(norm_benchmark, 'g', label='Benchmark')
    plt.plot(norm_optimal_portfolio, 'r', label='Optimal Portfolio')
    ax.legend(loc=2)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig('figure.png')

    avg_daily_ret, std_daily_ret, cum_ret = get_stats(benchmark)
    with open('p6_results.txt','w') as f:
        # f.write("benchmark",avg_daily_ret, std_daily_ret, cum_ret)
        print("benchmark",avg_daily_ret, std_daily_ret, cum_ret, file=f)
        avg_daily_ret, std_daily_ret, cum_ret = get_stats(optimal_portfolio)
        print("optimal", avg_daily_ret, std_daily_ret, cum_ret, file=f)
        f.close()

    df_trades = orders.to_frame()




    return df_trades


def get_stats(port_val):
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * daily_rets.mean() / std_daily_ret
    cum_ret = port_val.iloc[-1]/port_val.iloc[0] - 1.
    return avg_daily_ret.values[0], std_daily_ret.values[0], cum_ret.values[0]
