import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    '''
    Softmax 损失函数，最简单的实现（带循环）

    输入有D个维度， C个分类，我们操作在N个样本的 minibatches
    L = XW+reg*sum(W*W)
    Inputs:
    - W: (D,C)包含权重
    - X: (N,D)包含了一个minibatch的数据
    - y: (N,)包含了训练数据的标签;y[i] = c意思为X[i]的标签为C
      0<= c <C
    - reg: 正则化强度

    Returns 返回一个元组
    - loss: 一个float数
    - 关于权重W的梯度；一个数组shape和W一致
    '''
    # 初始化loss和gradient为0
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        # 数值稳定性问题 softmax上下各除以C（logC= -max）
        scores = scores - np.max(scores)
        sum_scores = np.sum(np.exp(scores))
        loss_i = -scores[y[i]] + np.log(sum_scores)
        loss += loss_i
        for j in range(num_classes):
            # dW[:, j] += X[i] * np.exp(scores[j]) / sum_scores
            dW[:, j] += X[i] * np.exp(scores[j]) / sum_scores
            if j == y[i]:
                dW[:, j] -= X[i]
    # 对批次求平均值并添加正则化项的导数。
    dW /= num_train
    dW += 2 * reg * W

    # 对批次求平均值并添加我们的正则化项。
    loss /= num_train
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW

# def softmax_loss_naive(W, X, y, reg):
#   """
#   Softmax loss function, naive implementation (with loops)
#   Inputs have dimension D, there are C classes, and we operate on minibatches
#   of N examples.
#   Inputs:
#   - W: A numpy array of shape (D, C) containing weights.
#   - X: A numpy array of shape (N, D) containing a minibatch of data.
#   - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#     that X[i] has label c, where 0 <= c < C.
#   - reg: (float) regularization strength
#   Returns a tuple of:
#   - loss as single float
#   - gradient with respect to weights W; an array of same shape as W
#   """
#   # Initialize the loss and gradient to zero.
#   loss = 0.0
#   dW = np.zeros_like(W)
#
#   #############################################################################
#   # TODO: Compute the softmax loss and its gradient using explicit loops.     #
#   # Store the loss in loss and the gradient in dW. If you are not careful     #
#   # here, it is easy to run into numeric instability. Don't forget the        #
#   # regularization!                                                           #
#   #############################################################################
#   num_classes = W.shape[1]
#   num_train = X.shape[0]
#
#   for i in range(num_train):
#      scores = X[i].dot(W)
#      shift_scores = scores - max(scores)
#      loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
#      loss += loss_i
#      for j in range(num_classes):
#          softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
#          if j == y[i]:
#              dW[:,j] += (-1 + softmax_output) *X[i]
#          else:
#              dW[:,j] += softmax_output *X[i]
#
#   loss /= num_train
#   loss +=  0.5* reg * np.sum(W * W)
#   dW = dW/num_train + reg* W
#   #pass
#   #############################################################################
#   #                          END OF YOUR CODE                                 #
#   #############################################################################
#
#   return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    # 计算分数和数字稳定性修复。
    scores = np.dot(X, W)
    scores = scores - np.max(scores, keepdims=True, axis=1)

    sum_scores = np.sum(np.exp(scores), axis=1)

    softmax_scores = np.exp(scores) / sum_scores.reshape(-1, 1)
    # Li = -fyi + log \sum(e^f_j)
    loss = -np.sum(scores[np.arange(num_train), y]) + np.sum(np.log(sum_scores))

    dS = softmax_scores.copy()
    dS[range(num_train), y] -= 1

    dW += np.dot(X.T, dS)
    dW /= num_train
    dW += 2 * reg * W

    loss /= num_train
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW