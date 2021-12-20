""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import InsaneLearner as it
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import matplotlib.pyplot as plt
from scipy import stats
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )

    alldata = np.genfromtxt(inf, delimiter=",")
    if sys.argv[1] == "Data/Istanbul.csv":
        alldata = alldata[1:, 1:]
    # Skip the date column and header row if we're working on Istanbul data
    data = alldata
    # df = pd.read_csv('Data/Istanbul.csv')
    # df = df.dropna()
    # df = df.drop(['date'], axis=1)
    # data = df.to_numpy()

  		  	   		   	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		   	 		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=False)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it  		  	   		   	 		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		   	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		   	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		   	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		   	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		   	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		   	 		  		  		    	 		 		   		 		  

    # #  DTLearner
    Xtrain = train_x
    Ytrain = train_y
    Xtest = test_x
    Ytest = test_y
    # learner = dt.DTLearner(leaf_size=1, verbose=False)  # constructor
    # learner.add_evidence(Xtrain, Ytrain)  # training step
    # Ypred = learner.query(Xtrain)  # query
    # corr = np.corrcoef(Ytrain,Ypred)[0, 1]
    # print(corr)
    # print(learner.author())
    #
    # # plt.scatter(Ytrain, Ypred)
    # # plt.savefig('50')
    #
    # #RTLearner
    # learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    # learner.add_evidence(Xtrain, Ytrain)  # training step
    # Ypred = learner.query(Xtest)  # query
    # coef = np.corrcoef(Ypred,Ytest)[0, 1]
    # print('RTLearner', coef)
    # print(learner.author())
    #
    # #BagLearner
    #
    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=10, boost=False, verbose=False)
    # learner.add_evidence(Xtrain, Ytrain)
    # Ypred = learner.query(Xtest)
    # coef = np.corrcoef(Ypred, Ytest)[0, 1]
    # print('BagLearner for RTLearner', coef)
    # print(learner.author())
    #
    # learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=10, boost=False, verbose=False)
    # learner.add_evidence(Xtrain, Ytrain)
    # Ypred = learner.query(Xtest)
    # coef = np.corrcoef(Ypred, Ytest)[0, 1]
    # print('BagLearner for DTLearner', coef)
    #
    #
    # learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=10, boost=False, verbose=False)
    # learner.add_evidence(Xtrain, Ytrain)
    # Y = learner.query(Xtest)
    # coef = np.corrcoef(Ypred, Ytest)[0, 1]
    # print('BagLearner for LinRegLearner', coef)
    #
    # # InsaneLerner
    #
    # learner = it.InsaneLearner(verbose=False)  # constructor
    # learner.add_evidence(Xtrain, Ytrain)  # training step
    # Ypred = learner.query(Xtest)  # query
    # coef = np.corrcoef(Ypred, Ytest)[0, 1]
    # print('InsaneLearner', coef)
    # print(learner.author())
    trainerr = []
    testerr = []
    lst = list(range(1,40,3))
    for size in lst:
        learner = dt.DTLearner(leaf_size=size, verbose=False)  # constructor
        learner.add_evidence(Xtrain, Ytrain)  # training step
        Ypred = learner.query(Xtrain)  # query
        trainerr.append(math.sqrt(np.square(np.subtract(Ypred, Ytrain)).mean()))
        Ypred = learner.query(Xtest)  # query
        testerr.append(math.sqrt(np.square(np.subtract(Ypred, Ytest)).mean()))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("Figure 1")
    ax.set_xlabel('Leaf Size')
    ax.set_ylabel("Root Mean Square Error")
    plt.plot(lst,trainerr, label='Training Error')
    plt.plot(lst,testerr, label='Testing Error')
    ax.legend(loc = 4)
    plt.tight_layout()
    plt.savefig('figure1.png')

    trainerr = []
    testerr = []
    lst = list(range(1, 40, 3))
    for size in lst:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": size}, bags=20, boost=False, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)  # training step
        Ypred = learner.query(Xtrain)  # query
        trainerr.append(math.sqrt(np.square(np.subtract(Ypred, Ytrain)).mean()))
        Ypred = learner.query(Xtest)  # query
        testerr.append(math.sqrt(np.square(np.subtract(Ypred, Ytest)).mean()))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("Figure 2")
    ax.set_xlabel('Leaf Size')
    ax.set_ylabel("Root Mean Square Error")
    plt.plot(lst, trainerr, label='Training Error')
    plt.plot(lst, testerr, label='Testing Error')
    ax.legend(loc=4)
    plt.tight_layout()
    plt.savefig('figure2.png')

    RTerr = []
    DTerr = []
    DTstd = []
    RTstd = []
    lst = list(range(1, 40, 3))
    for bag in lst:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=bag, boost=False, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)  # training steps
        Ypred = learner.query(Xtest)  # query
        slope, intercept, r_value, p_value, std_err = stats.linregress(Ypred, Ytest)
        DTerr.append(r_value)
        DTstd.append(std_err)
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=bag, boost=False, verbose=False)
        learner.add_evidence(Xtrain, Ytrain)  # training steps
        Ypred = learner.query(Xtest)  # query
        slope, intercept, r_value, p_value, std_err = stats.linregress(Ypred, Ytest)
        RTerr.append(r_value)
        RTstd.append(std_err)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("Figure 3")
    ax.set_xlabel('Bag Size')
    ax.set_ylabel("R-squared Correlation of Coefficient")
    plt.plot(lst, DTerr, label='Decision Tree')
    plt.plot(lst, RTerr, label='Random Tree')
    ax.legend(loc=4)
    plt.tight_layout()
    plt.savefig('figure3.png')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    ax.set_title("Figure 4")
    ax.set_xlabel('Bag Size')
    ax.set_ylabel("Standard Deviation Error")
    plt.plot(lst, DTstd, label='Decision Tree')
    plt.plot(lst, RTstd, label='Random Tree')
    ax.legend(loc=4)
    plt.tight_layout()
    plt.savefig('figure4.png')