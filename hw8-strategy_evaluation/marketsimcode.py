import datetime as dt
import os
from copy import deepcopy

import numpy as np

import pandas as pd
from util import get_data, plot_data

def author():
    return "sgorucu3"


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    orders_df = orders_df.sort_index()
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    symbols = orders_df['Symbol'].unique()
    portvals = get_data(symbols, pd.date_range(start_date, end_date))
    dfprices = portvals[symbols]  # remove SPY
    dateindex = portvals.index
    symbols = np.append(symbols, "CASH")
    zeros = np.zeros((dateindex.shape[0], len(symbols)))
    dftrades = pd.DataFrame(zeros, index=dateindex, columns=symbols)
    dfholdings = deepcopy(dftrades)
    dfholdings.iloc[0]["CASH"] = start_val
    yesterday = dfholdings.iloc[0]
    dfportval = pd.DataFrame(np.zeros(dateindex.shape[0]), index=dateindex, columns=["Total"])
    for date in dateindex:
        dfholdings.loc[date] = yesterday
        if date in orders_df.index:
            orders = orders_df.loc[[date]]
            for i in range(len(orders.index)):
                order = orders.iloc[i]
                symbol = order["Symbol"]
                price = dfprices.loc[date][symbol]
                size = order["Shares"]
                if order["Order"] == "SELL":
                    size *= -1
                dftrades.loc[date][symbol] += size
                dftrades.loc[date]["CASH"] -= size * price
                dftrades.loc[date]["CASH"] -= commission
                dftrades.loc[date]["CASH"] -= abs(size * price * impact)
        dfholdings.loc[date] += dftrades.loc[date]
        yesterday = dfholdings.loc[date]
        dfportval.loc[date]["Total"] = dfholdings.loc[date]["CASH"]
        for symbol in symbols:
            if symbol == "CASH":
                break
            dfportval.loc[date]["Total"] += dfholdings.loc[date][symbol] * dfprices.loc[date][symbol]

    return dfportval