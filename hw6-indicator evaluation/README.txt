There are 4 files:
-testproject.py
-TheoreticallyOptimalStrategy.py
-indicators.py
-marketsimcode.py

testproject.py is the main file and calls all the necessary function required to generate plots and to make calculations.

TheoriticallyOptimalStrategy.py:
This file has the testPolicy function. testPolicy function calculates the theoritically optimal strategy, create its holding. It also computes the holding values for the benchmark. Holding values are computed by calling marketsimcode. A benchmark to tos comparison plot is created here. Further, get_stats function computes average daily return, standard deviation of daily return and cumulative return. get_stats is call from testPolicy. testPolicy finally returns a one column dataframe that includes all the orders for the optimal strategy.

indicators.py:
This file computes and plots several technical indicators such as SMA, Momentum, MACD, Bollinger Bands, Stochastic Oscillator and Aroon Oscilator.

marketsimcode.py:
This file computes the holding values for a portfolio given start date, end date, ticker symbol and the orders dataframe.
