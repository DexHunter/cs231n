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
  - gradient with respect to weights W; an array of same shape as W (D,C)
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]#C, normally 10
  num_train = X.shape[0]#N, 500 in test case
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # this is a vector (10,) !
    #print scores
    correct_class_score = scores[y[i]] # this is a number!
    #print '--------------'
    #print correct_class_score
    #print 'i is ', i
    for j in xrange(num_classes):
      if j == y[i]:
        #print "right case", j
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # there are 9 times for each i when margin > 0
      if margin > 0:
          #print "j is ", j
          dW[:,y[i]] += -X[i,:]
          dW[:,j] += X[i,:]
          loss += margin

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
  dW /= num_train
  dW += 2 * reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  X (500, 3073)
  W (3073, 10)
  y (500,)
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  delta = 1
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y].reshape((num_train,1))
  margin = scores - correct_class_scores + delta
  margin[margin<0] = 0
  margin[np.arange(num_train), y] = 0 #avoid counting j=y[i]
  # margin shape (500, 10)
  #print margin
  loss = np.sum(margin)
  loss /= num_train

  loss += 0.5 * reg * np.sum(W * W)


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
  mask = np.zeros(margin.shape)
  mask[margin > 0] = 1

  incorrect_counts = np.sum(mask, axis=1)
  #print incorrect_counts
  mask[np.arange(num_train) , y] = -incorrect_counts


  dW = X.T.dot(mask)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
