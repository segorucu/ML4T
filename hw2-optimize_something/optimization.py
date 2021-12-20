""""""
"""MC1-P2: Optimize a portfolio.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Seyhan Emre Gorucu (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: sgorucu3 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903658809  (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from util import get_data, plot_data
from scipy.optimize import minimize
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# This is the function that will be tested by the autograder  		  	   		   	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality

def constraint(x):
    return sum(x) - 1


class FundCalc(object):

    def __init__(self,prices):
        self.prices = prices

    def calc_return(self):
        self.cr = self.fund[-1]/self.fund[0] - 1.
        self.dr = self.fund[1:] / self.fund[:-1].values - 1.
        self.adr = self.dr.mean()
        self.sddr = self.dr.std()

    def clcsharpe(self, allocs):
        self.allocs = allocs
        self.fund = self.prices.dot(self.allocs)
        self.calc_return()
        self.sharpe = self.adr / self.sddr
        self.sharpe *= (252 ** 0.5)
        return - self.sharpe


def calc_sharpe(allocs, prices):
    fund = prices.dot(allocs)
    # dr = fund.copy()
    dr = fund[1:] / fund[:-1].values - 1.
    # dr.ix[0] = 0
    adr = dr.mean()
    std = dr.std()
    sharpe = adr / std
    sharpe *= (252**0.5)

    return -sharpe


def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 		  		  		    	 		 		   		 		  
    statistics.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = prices_all / prices_all.iloc[0]
    prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  

    # find the allocations for the optimal portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case
    n = len(syms)
    allocs = np.zeros(n)
    allocs[:] = 1. / n

    bound = []
    for i in range(n):
        bound.append([0, 1.])
    opt = minimize(calc_sharpe, allocs, args =(prices), method="SLSQP", \
                   bounds=bound, constraints={'type': 'eq', 'fun': constraint})
    allocs = opt.x

    obj = FundCalc(prices)
    obj.clcsharpe(allocs)
    cr = obj.cr
    dr = obj.dr
    adr = obj.adr
    sddr = obj.sddr
    sr = obj.sharpe

    # Get daily portfolio value  		  	   		   	 		  		  		    	 		 		   		 		  
    port_val = obj.fund # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		   	 		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here

        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.set_title("Daily Portfolio Value and SPY")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (Normalized)")
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        plt.plot(df_temp)
        ax.legend(["Portfolio", "SPY"])
        #ax.set_xlim(left=df_temp.index[0], right=df_temp.index[-1])
        #plt.xticks(rotation=80)
        #plt.show()
        plt.savefig('figure.png')


    return (allocs, cr, adr, sddr, sr)


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]

    # Assess the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )

    # Print statistics  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		   	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		   	 		  		  		    	 		 		   		 		  
    test_code()
