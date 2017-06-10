import numpy as np
from random import shuffle

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
  dW = np.zeros_like(W) #(3073,10)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] #500, shape of X (500, 3073)
  num_class = W.shape[1]
  for i in xrange(num_train):
      fi = X[i,:].dot(W) #(10,)
      fi -= np.max(fi) #avoid numerical instability according to note
      #print fi.shape
      pi = np.exp(fi[y[i]]) / np.sum(np.exp(fi))
      loss += -np.log(pi)
      for j in xrange(num_class):
          p = np.exp(fi[j]) / np.sum(np.exp(fi))
          dW[:, j] += (p - (j==y[i]))*X[i,:] #minus one when y[i]==j

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]

  f = X.dot(W) #array store all scores
  f -= np.max(f, axis=1, keepdims=True)
  fi = f[np.arange(num_train), y].ravel() #correct class score
  pi = np.exp(fi) / np.sum(np.exp(f), axis=1) #column-wise summation
  loss = np.sum(-np.log(pi))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  p = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)
  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1 #correct class set to 1
  p -= ind
  dW = X.T.dot(p)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

