from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for i in range(self.num_layers):
          if i == 0:  
            w = np.random.normal(0.0, weight_scale, size=(input_dim, hidden_dims[i]))
            b = np.zeros(hidden_dims[i])
          elif i == self.num_layers-1:
            w = np.random.normal(0.0, weight_scale, size=(hidden_dims[i-1], num_classes))
            b = np.zeros(num_classes)
          else:
            w = np.random.normal(0.0, weight_scale, size=(hidden_dims[i-1], hidden_dims[i]))
            b = np.zeros(hidden_dims[i])

          key_name = f"W{i+1}"
          key_name2 = f"b{i+1}"
          self.params[key_name]=w
          self.params[key_name2]=b
        
        if self.normalization=="batchnorm" or self.normalization=="layernorm":
          for j in range(len(hidden_dims)):
            # print("lol")
            gamma=np.ones(hidden_dims[j])
            beta=np.zeros(hidden_dims[j])
            key_name = f"gamma{j+1}"
            key_name2 = f"beta{j+1}"
            self.params[key_name]=gamma
            self.params[key_name2]=beta


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            # print(k)
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = []

        for i in range(self.num_layers):
            key_name = f"W{i+1}"
            key_name2 = f"b{i+1}"
            w = self.params[key_name]
            b = self.params[key_name2]

            if i == 0:
                # First hidden layer
                hid, cache1 = affine_forward(X, w, b)  # Affine transformation with input X
                
                if self.normalization == "batchnorm":  # Check if BatchNorm is enabled
                    hid, cache2 = batchnorm_forward(hid, self.params["gamma1"], self.params["beta1"], self.bn_params[i])
                elif self.normalization == "layernorm":
                    hid, cache2 = layernorm_forward(hid, self.params["gamma1"], self.params["beta1"], self.bn_params[i])
                else:
                    cache2 = None  # If not using BatchNorm, store None for cache2

                hid, cache3 = relu_forward(hid)  # ReLU activation
                if self.use_dropout:
                  # print("lol1111111111111111111111111111111")
                  # dropout_param = self.dropout_param if self.use_dropout else None
                  # print(self.dropout_param)
                  # print(self.use_dropout)
                  hid,cache4= dropout_forward(hid,self.dropout_param)
                else:
                  cache4=None
                c = (cache1, cache2, cache3,cache4)  # Store all caches
                caches.append(c)
                continue

            if i == self.num_layers - 1:
                # Output layer (no BatchNorm or ReLU)
                scores, cache = affine_forward(hid, w, b)
                caches.append(cache)
                break

            # Hidden layers
            hid, cache1 = affine_forward(hid, w, b)

            if self.normalization == "batchnorm":  # Check if BatchNorm is enabled
                hid, cache2 = batchnorm_forward(hid, self.params[f"gamma{i+1}"], self.params[f"beta{i+1}"], self.bn_params[i])
            elif self.normalization == "layernorm":
                hid, cache2 = layernorm_forward(hid, self.params["gamma1"], self.params["beta1"], self.bn_params[i])
            else:
                cache2 = None  # If not using BatchNorm, store None for cache2

            hid, cache3 = relu_forward(hid)
            if self.use_dropout:
              # print("lol2")
              hid,cache4= dropout_forward(hid,self.dropout_param)
            else:
                cache4=None
            c = (cache1, cache2, cache3,cache4)  # Store all caches
            caches.append(c)

          # cashes.append(cache)

        # print("llll")


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)

        for j in range(self.num_layers):
            # Define the key names for this layer's weights and biases
            key_name = f"W{self.num_layers-j}"
            key_name2 = f"b{self.num_layers-j}"

            # Retrieve caches for the previous affine layer
            cache = caches[self.num_layers-j-1]

            if j == 0:
                # Backprop through the last affine layer (output layer)
                dhidden, dW, db = affine_backward(dscores, cache)
                grads[key_name] = dW + self.reg * self.params[key_name]  # Weight gradients with L2 regularization
                grads[key_name2] = db  # Bias gradients
                continue

            # Backprop through ReLU for hidden layers (conditionally using BatchNorm)
            
            if self.normalization == "batchnorm":
              if self.use_dropout:
                dhidden, dW1, db1, dgamma, dbeta = affine_batchnorm_relu_dropout_backward(dhidden, (cache))
                grads[f"gamma{self.num_layers-j}"] = dgamma
                grads[f"beta{self.num_layers-j}"] = dbeta
              else:
                # Perform backpropagation through the affine -> batchnorm -> relu layer
                dhidden, dW1, db1, dgamma, dbeta = affine_batchnorm_relu_backward(dhidden, (cache))
                grads[f"gamma{self.num_layers-j}"] = dgamma
                grads[f"beta{self.num_layers-j}"] = dbeta
            if self.normalization == "layernorm":
              if self.use_dropout:
                dhidden, dW1, db1, dgamma, dbeta = affine_layernorm_relu_dropout_backward(dhidden, (cache))
                grads[f"gamma{self.num_layers-j}"] = dgamma
                grads[f"beta{self.num_layers-j}"] = dbeta
              else:
                dhidden, dW1, db1, dgamma, dbeta = affine_layernorm_relu_backward(dhidden, (cache))
                grads[f"gamma{self.num_layers-j}"] = dgamma
                grads[f"beta{self.num_layers-j}"] = dbeta

            else:
              if self.use_dropout:
                dhidden, dW1, db1 = affine_relu_dropout_backward(dhidden, (cache))
              else:
                # Perform backpropagation through the affine -> relu layer (no BatchNorm)
                dhidden, dW1, db1 = affine_relu_backward(dhidden, (cache))

            grads[key_name] = dW1 + self.reg * self.params[key_name]  # Weight gradients with L2 regularization
            grads[key_name2] = db1

        # Compute L2 regularization for all layers
        for i in range(self.num_layers):
            key_name = f"W{i+1}"
            loss += 0.5 * self.reg * np.sum(self.params[key_name] ** 2)


        # current_cache = cashes.pop()
        # dx, dw, db = affine_backward(dscores, current_cache)
        # grads[f"W{self.num_layers}"] = dw + self.reg * self.params[f"W{self.num_layers}"]
        # grads[f"b{self.num_layers}"] = db

        # # Backprop through the rest of the affine + ReLU layers
        # for i in reversed(range(1, self.num_layers)):
        #     current_cache = cashes.pop()
        #     dx, dw, db = affine_relu_backward(dx, current_cache)
            
        #     grads[f"W{i}"] = dw + self.reg * self.params[f"W{i}"]
        #     grads[f"b{i}"] = db

        # for i in range(self.num_layers):
        #     key_name = f"W{i+1}"
        #     loss += 0.5 * self.reg * np.sum(self.params[key_name] ** 2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
