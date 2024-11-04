from builtins import range
from re import S
import numpy as np
from random import sample, shuffle
from past.builtins import xrange
import math


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
      sample_weight=W[i]
      for j in range(W.shape[1]):
        # dW[:j]=np.power(W[i][j],2)
        sample_weight[j]=math.exp(sample_weight[j])
      sum_all=0
      for s in range(W.shape[1]):
        sum_all+=sample_weight[s]
      for l in range(W.shape[1]):
        sample_weight[l]/=sum_all
      loss+= -math.log(sample_weight[y[i]])
      for j in range(W.shape[1]):
        if j == y[i]:
            # For the correct class: (P_j - 1) * X[i]
            dW[:, j] += (sample_weight[j] - 1) * X[i]
        else:
            # For incorrect classes: P_j * X[i]
            dW[:, j] += sample_weight[j] * X[i]
      

    # for i in range(X.shape[0]):
    #   # Compute scores for all classes
    #   scores = np.dot(X[i], W)
      
    #   # Normalize scores for numerical stability
    #   scores -= np.max(scores)
      
    #   # Compute softmax probabilities
    #   exp_scores = np.exp(scores)
    #   probs = exp_scores / np.sum(exp_scores)  # Normalized probabilities
      
    #   # Compute the loss for the correct class
    #   correct_class_prob = probs[y[i]]  # Probability of the correct class
    #   loss += -np.log(correct_class_prob)  # Cross-entropy loss for the correct class
      
    #   # Compute the gradient for this example
    #   for j in range(W.shape[1]):
    #       if j == y[i]:
    #           # For the correct class: (P_j - 1) * X[i]
    #           dW[:, j] += (probs[j] - 1) * X[i]
    #       else:
    #           # For incorrect classes: P_j * X[i]
    #           dW[:, j] += probs[j] * X[i]
      
    loss/=X.shape[0]
    loss+=reg*np.sum(W*W)
    dW/=X.shape[0]
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


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

    scores=np.dot(X,W)
    scores -= np.max(scores, axis=1, keepdims=True)
    scores=np.exp(scores)
    sum_all=np.sum(scores, axis=1, keepdims=True)
    scores/=sum_all
    correct_class_scores = scores[np.arange(X.shape[0]), y]
    loss=-np.sum(np.log(correct_class_scores))
    loss/=X.shape[0]

    grad = scores
    grad[np.arange(X.shape[0]), y] -= 1
    grad /= X.shape[0]
    
    dW = np.dot(X.T, grad)
    
    # Add regularization to the gradient
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
