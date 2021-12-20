import pandas as pd
import numpy as np
np.set_printoptions(precision=1)
import random


class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):

        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = []
        self.bracket = None

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
        num_features = len(data_x[1])
        self.bracket = 1.0 / num_features
        tree = self.build_tree(data_x, data_y)
        self.tree = tree

        return

    def build_tree(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size and data_x.shape[0] != 1:
            return [-1, data_y.mean(), -1, -1]
        if data_x.shape[0] == 1 or data_x.shape[0] <= self.leaf_size:
            #(feature index, separation value, left node, right node)
            # self.tree.append([])
            return [-1, data_y[0], -1, -1]
        if np.max(data_y) == np.min(data_y):
            # self.tree.append([])
            return [-1, data_y[0], -1, -1]
        # decide best feature value to split on
        num = random.random()
        feature_index = num // self.bracket
        feature_index = int(feature_index)
        if min(data_x[:, feature_index]) == max(data_x[:, feature_index]):
            return [-1, data_y.mean(), -1, -1]

        SplitVal = np.median(data_x[:,feature_index])


        if self.verbose:
            ind = data_x[:,feature_index] == SplitVal
            if data_x[ind].shape[0] >= 2:
                print('There are ', data_x[ind].shape[0], 'medians.' )
        index = data_x[:,feature_index] <= SplitVal
        if np.all(index) or np.all(index == False):
            SplitVal = np.mean(data_x[:, feature_index])
            index = data_x[:, feature_index] <= SplitVal
        if self.verbose:
            if np.all(index) or  np.all(index == False):
                print('all true left tree')
        left_tree = self.build_tree(data_x[index],data_y[index])
        index = data_x[:, feature_index] > SplitVal
        if np.all(index) or  np.all(index == False):
            SplitVal = np.mean(data_x[:, feature_index])
            index = data_x[:, feature_index] > SplitVal
        if self.verbose:
            if np.all(index) or  np.all(index == False):
                print('all true right_tree')
        right_tree = self.build_tree(data_x[index], data_y[index])

        try:
            nrows = left_tree.shape[0]
        except:
            nrows = 1
        root = [feature_index, SplitVal, 1, nrows+1]

        return np.vstack((root, left_tree, right_tree))

    def query(self,Xtest):

        Ytest = []
        for row in range(len(Xtest)):
            Xval = Xtest[row,:]
            try:
                feature_index = self.tree[0][0]
                feature_index = int(feature_index)
                SplitVal = self.tree[0][1]
            except:
                SplitVal = self.tree[1]
                feature_index = -1
            node = 0
            while feature_index >= 0:
                if Xval[feature_index] <= SplitVal:
                    node += self.tree[node][2]
                else:
                    node += self.tree[node][3]
                node = int(node)
                feature_index = self.tree[node][0]
                feature_index = int(feature_index)
                SplitVal = self.tree[node][1]


            Yval = SplitVal
            Ytest.append(Yval)
        Ytest = np.array(Ytest)

        return Ytest

def author():
    return "sgorucu3"