from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
from past.builtins import xrange

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3,
              reg=1e-5, num_iters=100, batch_size=200,
              verbose=False):

        """
        训练线性分类器用随机梯度下降

        Inputs:
        - X: (N,D)包含训练数据；N是训练样本的数目，D是每条数据的维度
        - y: (N,)训练标签；y[i] = c值X[i]的标签为 c,0<= c < C
        - learning_rate: (float)学习率
        - reg: (float)正则项的强度
        - num_iters: (integer) 优化时要采取的steps 迭代的次数
        - batch_size: (integer)  每个step要使用的训练样本数。
        - verbose: (boolean) 如果是true，打印优化过程。

        Outputs:
        list:包含每次训练迭代中loss。
        """

        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )
        if self.W is None:
            # 延迟初始化W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 随机梯度下降优化 w
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            batch_idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # pass
            self.W += -learning_rate * grad
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def loss(self, X_batch, y_batch, reg):
        """
        计算损失和它的偏导
        子类将会覆写这个方法

        Inputs:
        - X_batch: (N,D) 包含了一个minibatch的数据,N条数据;每个数据D维度
        - y_batch: (N,) minibatch的标签
        - reg: (float) 正则项的强度

        Returns： 一个元组
        - loss: 一个float值
        - dW: W的梯度
        """
        pass

    def predict(self, X):
        """
        使用此线性分类器的训练权重来预测标签。
        """
        y_pred = np.zeros_like(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        pred_scores = np.dot(X, self.W)
        y_pred = np.argmax(pred_scores, axis=1)

        return y_pred

class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)