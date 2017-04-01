import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] #C
  num_train = X.shape[0] #N
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        # skip for the true class to only loop over incorrect classes
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # calculate gradient using Calculus
        # a.k.a. analytic gradient
        # margin > 0 means the indicator function is true
        # differentiate with respect to weight
        # iterate over all the class
        x_t = X.T
        #print x_t.shape
        dW[:,j] = dW[:,j] + x_t[:,i]
        # subtract the true class
        dW[:,y[i]] = dW[:,y[i]] - x_t[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW / num_train
  # regularization?
  dW += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  C = W.shape[1] #num_classes
  N = X.shape[0] #num_train
  scores = X.dot(W)
  delta = 1
  margin = scores - scores[np.arange(N),y][:,None] + delta

  mask0 = np.zeros((N,C),dtype=bool)
  mask0[np.arange(N),y] = 1
  mask = (margin <= 0) | mask0
  margin[mask] = 0 # if <=0 then set to 0

  loss = margin.sum() / N

  # Add regularization to the loss
  #loss += 0.5 * reg * np.sum(W * W)
  loss += 0.5 * reg * float(np.tensordot(W,W,axes=((0,1),(0,1))))
  # according to http://stackoverflow.com/questions/42971039/how-to-vectorize-loss-in-svm
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask[np.arange(N),y] = 0
  dW = (X.T).dot(mask)
  dW /= N
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
