import numpy as np
from random import shuffle
from past.builtins import xrange

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
  for i in range(X.shape[0]):
    scores = X[i].dot(W)
        
    idx_max = np.argmax(scores)
    s_max = scores[idx_max]
    scores -= s_max    # shift for numerical stability
    
    temp = np.exp(scores)
    summation = np.sum(temp)
    loss += (- scores[y[i]] + np.log(summation))
    
    # computing gradients
    # (1) an explicit version:
#     for j in range(W.shape[1]):
#         if j == y[i]:
#             dW[:, j] -= X[i]
#             dW[:, idx_max] -= (-X[i])
        
#             dW[:, j] += (1 / summation) * temp[j] * X[i]
#             dW[:, idx_max] += (1 / summation) * temp[j] * (-X[i])
#         elif j == idx_max:
#             dW[:, j] += 0    # X[i] + (-X[i]) = 0
#         else:
#             dW[:, j] += (1 / summation) * temp[j] * X[i]
#             dW[:, idx_max] += (1 / summation) * temp[j] * (-X[i])
    
    # (2) a more concise version:
    softmax_scores = temp / summation
    for j in range(W.shape[1]):
        if j == y[i]:
            dW[:, j] += (-1 + softmax_scores[j]) * X[i]
        else:
            dW[:, j] += softmax_scores[j] * X[i]
    
  loss /= X.shape[0]
  dW /= X.shape[0]
    
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

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

  scores = X.dot(W)                        # (1)
  s_max = np.max(scores, 1)
  scores -= s_max.reshape((-1, 1))
  
  s_y = scores[np.arange(num_train), y]    # (2)
  s_exp = np.exp(scores)                   # (3)
  summation = np.sum(s_exp, 1)             # (4)
  data_loss = np.sum(-s_y + np.log(summation)) / num_train    # (5)
    
  loss = data_loss + reg * np.sum(W * W)
  
  # computing gradients: staged computation!!
  dsy = -1 / num_train                            # (5)
  dsummation = (1 / summation) / num_train        # (5)

  dsexp = 1 * dsummation                          # (4)
    
  dscores = s_exp * dsexp.reshape((-1, 1))        # (3)
  dscores[np.arange(num_train), y] += 1 * dsy     # (2)
  
  dW += X.T.dot(dscores)                          # (1)
  
  dW += 2 * reg * W 


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

