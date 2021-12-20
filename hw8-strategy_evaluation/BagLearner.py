import pandas as pd
import numpy as np
np.set_printoptions(precision=1)


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose=False):

        self.learners = []
        #kwargs = {"k": 10}
        for bag in range(0, bags):
            self.learners.append(learner(**kwargs))

        self.verbose = verbose
        self.bags = bags

        return

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "sgorucu3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        datasize = len(data_x)
        for bag in range(self.bags):
            index = np.random.randint(0, datasize, datasize)
            data_xn = data_x[index,:]
            data_yn = data_y[index]
            self.learners[bag].add_evidence(data_xn, data_yn)

        return

    def query(self,Xtest):

        Ytest = np.zeros(len(Xtest))
        for bag in range(self.bags):
            Ytest = Ytest + self.learners[bag].query(Xtest)
        Ytest = Ytest / self.bags

        return Ytest

def author():
    return "sgorucu3"