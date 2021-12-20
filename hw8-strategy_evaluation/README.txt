This file provides the definitions of all the files inside and explains how to run them:

ManualStrategy.py: A file that the determines the buy, sell trade decisions based on technical analysis such as bollinger bands, sma(50)/sma(200), macd, Aroon indicator. Based on manually determined criteria, we make trade decisions.

marketsim.py: Once, trade decisions are made, a portfolio value can be calculated based on initial cash, start date and end date.

StrategyLearner.py: Trade decisions are made based on a random forest tree. add_evidence trains the in-sample data based on technical indicator values for each date. TestPolicy predicts, long, short, cash decision based on the trained model.

RTLearner.py: A random tree learner algorithm.

BagLearner.py: A bag learner consisting of several random trees.

testproject.py: This is  the code where all the necessary files are run.

indicators.py: This file calculates several technical indicators such as bollinger bands, macd, Aroon etc.

experiment1.py: This file plots a figure where benchmark, manual strategy and machine learning strategy are compared. Benchmark is buy 1000 shares of JPM and hold until the end date. Manual Strategy uses ManualStrategy.py to make buy, sell decisions. Machine learning strategy uses StrategyLearner tomake but sell decisions.

experiment2.py: This file plots a figures where different impact values are compared regarding its impact on the portfolio value. StrategyLearner is called for different impact values and the portfolio value is plotted for each scenario.

