import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # multiclass svm loss
  loss = 0.0
  for i in xrange(num_train):    # loop over training samples
    num_positive_margin = 0    # count number of margins greater than zero for each training sample
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):    # loop over classes for each training sample
      if j == y[i]:
        continue
        
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:    # an important condition: since W take effects on Loss only when margin>0
        loss += margin
        num_positive_margin += 1
        dW[:, j] += X[i]
        
    dW[:, y[i]] -= X[i] * num_positive_margin
    
  # according to computational graph to compute gradients
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


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
  scores = X.dot(W)    # N x C
  num_train = X.shape[0]
  rows = np.arange(num_train)

  hinge = np.maximum(scores - scores[rows, y].reshape((-1, 1)) + 1, 0)    # delta = 1
  hinge[rows, y] = 0    # true class 
  Li = np.sum(hinge, 1)
  data_loss = np.sum(Li) / num_train
    
  reg_loss = reg * np.sum(W * W)    # regularization loss

  loss = data_loss + reg_loss

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
  pos_mask = (hinge > 0).astype(float)    # mask of positive margins
  num_pos = pos_mask.sum(1)    # count how many times margins>0 for each image(i.e. per row)
  pos_mask[rows, y] = -num_pos    # negative for the weights of the true class 
  
  dW = X.T.dot(pos_mask)
  dW /= num_train
  dW += 2 * reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
