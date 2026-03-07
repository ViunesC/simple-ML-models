import numpy as np

from typing import Literal
from MLP.MyTensor import MyTensor


class MyNeuron:
    def __init__(
        self, n_feats, act_func: Literal["tanh", "linear", "sigmoid", "relu"]
    ) -> None:
        if act_func != "tanh":
            raise ValueError(
                "Activation function not supported yet. Plese use tanh at this time"
            )

        self.weights = [MyTensor(np.random.uniform(-1.0, 1.0)) for _ in range(n_feats)]
        self.bias = MyTensor(np.random.uniform(-1.0, 1.0))
        self.act_func = act_func

    def __call__(self, x):
        # forward pass on single neuron
        # y = wx + b
        linear_out = np.sum([w * x_i for w, x_i in zip(self.weights, x)]) + self.bias
        if self.act_func == "tanh":
            output = linear_out.tanh()
        else:
            raise ValueError(
                "Activation function not supported yet. Plese use tanh at this time"
            )

        return output

    def parameters(self):
        return self.weights + [self.bias]


class MyLayer:
    def __init__(
        self, n_feats, n_outs, act_func: Literal["tanh", "linear", "sigmoid", "relu"]
    ) -> None:
        """n_outs: number of neuron in the layer"""
        self.neurons = np.array([MyNeuron(n_feats, act_func) for _ in range(n_outs)])

    def __call__(self, x):
        return np.array([neuron(x) for neuron in self.neurons])

    def parameters(self):
        return np.array([p for neuron in self.neurons for p in neuron.parameters()])


class MyNeuralNet:
    def __init__(
        self,
        n_features: int,
        n_outputs: list,
        act_func: Literal["tanh", "linear", "sigmoid", "relu"] = "tanh",
    ) -> None:
        self.layers = [MyLayer(n_features, n_out, act_func) for n_out in n_outputs]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _parameters(self):
        return np.array([p for layer in self.layers for p in layer.parameters()])

    def fit(self, X: np.ndarray, y: np.ndarray, n_iters=1000, lr=1e-2, loss_type="mse", regularization="none"):
        """
            Train the neural network with given training set.

            :param X: numpy array containing training input set
            :param y: numpy array containing training output set
            :param n_iters: number of iterations for training
            :param lr: learning rate of training
            :param loss_type: type of loss used for training, 'mse' by default
            :param regularization: type of regularization applied during training, 'none' by default
        """
        for i in range(n_iters):
            # forward pass
            ypreds = np.array([self(x) for x in X])

            # calculate loss (we use mean square error here for simplicity)
            loss = np.sum([np.pow(y_p - y_act, 2) for y_act, y_p in zip(y, ypreds)])

            # backward propagation
            loss.backward()
            for p in self._parameters():
                # gradient descent
                p.data -= lr * p.grad
                p.grad = 0.0

            print(i + 1, loss.data)
        
        def predict(X):
            """
                Predict the outputs of given inputs.

                :param X: numpy array containing input set
            """
            ypreds = np.array([self(x) for x in X])
            return ypreds
