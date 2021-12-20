""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import datetime as dt
import random

import pandas as pd
import util as ut
import numpy as np
from indicators import *
import BagLearner as bl
import RTLearner as rt
# import DTLearner as dtl


class StrategyLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    def author(self):
        return "sgorucu3"
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.N = 6

    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        # add your code to do learning here  		  	   		   	 		  		  		    	 		 		   		 		  

        # example usage of the old backward compatible util function  		  	   		   	 		  		  		    	 		 		   		 		  
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        dfprices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:
            print(dfprices)

        # example use with new colname  		  	   		   	 		  		  		    	 		 		   		 		  
        # volume_all = ut.get_data(
        #     syms, dates, colname="Volume"
        # )  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        # if self.verbose:
        #     print(volume)

        Aroon = aroon_oscillator(dfprices)
        # K, D = stochastic_oscillator(symbol, sd, ed)
        bbp = bollinger(dfprices)
        smar = sma(dfprices)
        # mom = momentum(dfprices)
        macdrat = macd(dfprices)

        trainx = pd.concat((bbp, macdrat, Aroon),axis=1)
        trainx.columns = ["bbp", "macdrat", "Aroon"]
        trainx = trainx.fillna(0)
        trainx.drop(trainx.tail(self.N).index, inplace=True)
        trainx = trainx.to_numpy()

        yval = dfprices.values
        trainy = yval[self.N:] / yval[:-self.N] - 1
        Ybuy = 0.001 + self.impact + self.commission * 0.001
        Ysell = -0.001 - self.impact - self.commission * 0.001
        trainy[trainy >= Ybuy] = 1
        trainy[(trainy > Ysell) & (trainy < Ybuy)] = 0
        trainy[trainy < Ysell] = -1

        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=20, boost=False, verbose=False)
        self.learner.add_evidence(trainx, trainy)
        return

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        # here we build a fake set of trades  		  	   		   	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        dfprices = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # if self.verbose:
        #     print(type(trades))  # it better be a DataFrame!
        # if self.verbose:
        #     print(trades)
        # if self.verbose:
        #     print(prices_all)

        Aroon = aroon_oscillator(dfprices)
        # K, D = stochastic_oscillator(symbol, sd, ed)
        bbp = bollinger(dfprices)
        # smar = sma(dfprices)
        # mom = momentum(dfprices)
        macdrat = macd(dfprices)

        testx = pd.concat((bbp, macdrat, Aroon), axis=1)
        testx.columns = ["bbp", "macdrat", "Aroon"]
        testx = testx.fillna(0)
        # testx.drop(testx.tail(self.N).index, inplace=True)
        testx = testx.to_numpy()

        testy = self.learner.query(testx)
        testy[testy >= 0.5] = 1
        testy[(testy < 0.5) & (testy > -0.5)] = 0
        testy[testy < -0.5] = -1
        trades = pd.DataFrame(np.zeros(len(testy)), index=dfprices.index)
        holdings = 0
        for i, date in enumerate(dfprices.index):
            if testy[i] == 0 or i == dfprices.shape[0] - 1:
                trade = -holdings
                trades.iloc[i] = trade
                holdings = 0
            elif testy[i] == 1:
                trade = 1000 - holdings
                trades.iloc[i] = trade
                holdings = 1000
            elif testy[i] == -1:
                trade = - 1000 - holdings
                trades.iloc[i] = trade
                holdings = -1000
        return trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")

def author():
    return "sgorucu3"
