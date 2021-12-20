import numpy as np, BagLearner as bl, LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.ilearner = []
        for i in range(20):
            self.ilearner.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False))
    def author(self):
        return "sgorucu3"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for bag in range(20):
            index = np.random.randint(0, len(data_x), len(data_x))
            data_xn = data_x[index,:]
            data_yn = data_y[index]
            self.ilearner[bag].add_evidence(data_xn, data_yn)
    def query(self,Xtest):
        Ytest = np.zeros(len(Xtest))
        for bag in range(20):
            Ytest += self.ilearner[bag].query(Xtest)
        Ytest /= 20
        return Ytest

