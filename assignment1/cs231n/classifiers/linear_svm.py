from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    结构化SVM损失函数,朴素实现

    输入有D个维度， C个分类，我们操作在N个样本的 minibatches
    L = XW
    Inputs:
    - W: (D,C)包含权重
    - X: (N,D)包含了一个minibatch的数据
    - y: (N,)包含了训练数据的标签;y[i] = c意思为X[i]的标签为C
      0<= c <C
    - reg: 正则化强度

    Returns 返回一个元组
    - loss: 一个float数
    - 关于权重W的梯度；一个数组shape和W一致
    """

    dW = np.zeros(W.shape)  # 初始化梯度为0

    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]     # 为不正确的计算梯度变化
                dW[:, y[i]] -= X[i]
    # 现在损失是整个训练集的，但是我们想要是一个均值，所以除以num_train
    loss /= num_train
    dW /= num_train
    # 添加正则项为loss
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W       # 正则化项梯度
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    向量化实现SVM

        L = XW
    Inputs:
    - W: (D,C)包含权重
    - X: (N,D)包含了一个minibatch的数据
    - y: (N,)包含了训练数据的标签;y[i] = c意思为X[i]的标签为C
      0<= c <C
    - reg: 正则化强度

    Returns 返回一个元组
    - loss: 一个float数
    - 关于权重W的梯度；一个数组shape和W一致
    输入输出与svm_loss_naive相同
    """

    loss = 0.0
    dW = np.zeros(W.shape)  # 初始化梯度为0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    delta = 1.0

    scores = X.dot(W)    # (N,C)     #每一行代表每个类别的分别得分
    correct_class_scores = scores[range(num_train) , list(y)].reshape(-1, 1) # (N,1)
    margins = np.maximum(0, scores - correct_class_scores + delta)  #maximum逐位比较大小
    margins[range(num_train), list(y)] = 0      # 把正确分类的损失置为0

    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1                   # 错误分类进行梯度计算
    coeff_mat[range(num_train), list(y)] = 0     # 正确分类

    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)   # (N,C)

    dW = (X.T).dot(coeff_mat)           # dW(D,C)
    dW = dW / num_train + reg * W

    return loss, dW

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW