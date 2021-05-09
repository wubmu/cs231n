from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ knn分类器带有L2距离"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练模型，对于knn来说就是记住所有的训练数据

        Inputs:
        - X: 一个维度为(num_train,D)的numpy数组,包含了num_train条训练数据，每条数据
          是D维。
        - y: 一个维度为(num_train,)的numpy数组，包含了每条训练样本的标签，y[i]对应
          x[i]的label。
        """
        self.X_train = X
        self.Y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用这个分类器，预测所有测试数据

        Inputs:
        - X: (num_test,D)的numpy数组，由num_test条测试数据组成，每条测试数据为D维
        - k: 为预测标签投票的最近邻居的个数
        - num_loops: 决定计算训练数据和测试数据之间距离的实现方式
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # test矩阵平法 (p-q)^2 = p^2 + q^2 - 2pq
        te_2 = (X ** 2).sum(axis=1, keepdims=True)
        tr_2 = (self.X_train ** 2).sum(axis=1, keepdims=True)

        te_tr = self.X_train.dot(self.X_train.T)
        dists = np.sqrt(te_2 + tr_2 - 2 *te_tr)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i, :] = np.sqrt(np.sum(X[i] - self.X_train) ** 2, axis=1)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[i]) ** 2))
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def predict_labels(self, dists, k):
        """
        给一个距离矩阵训练样本与测试样本之间，预测标签为每一个测试样本

        Inputs：
        - disks: 一个(num_test,num_train)的数组,dists[i, j]
          代表第i个测试样本与第j个的训练样本的距离

        Returns:
        - y: (num_test,)的数组,包含每个测试样本的预测标签
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            closest_y = self.Y_train[np.argsort(dists[i])][:k]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            y_pred[i] = np.argmax(np.bincount(closest_y))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
