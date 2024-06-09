import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Optimizers
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients[i]
        return weights

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, gradients):
        if self.m is None:
            self.m = [np.zeros_like(w) for w in weights]
            self.v = [np.zeros_like(w) for w in weights]

        self.t += 1
        for i in range(len(weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weights

class NeuralNetwork:
    def __init__(self, shape, activations='sigmoid', use_batch_norm=False, l2_lambda=0.01, dropout_rate=0.0, optimizer='sgd', learning_rate=0.01):
        self.size = len(shape)
        self.shape = shape
        self.weights = []
        self.biases = []
        self.use_batch_norm = use_batch_norm
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.activations = activations if isinstance(activations, list) else [activations] * (self.size - 1)
        self.optimizer = self._init_optimizer(optimizer, learning_rate)

        self._initialize_weights()

        if self.use_batch_norm:
            self._initialize_batch_norm_params()

    def _init_optimizer(self, optimizer, learning_rate):
        """Initializes the optimizer."""
        if optimizer == 'sgd':
            return SGD(learning_rate)
        elif optimizer == 'adam':
            return Adam(learning_rate)
        else:
            raise ValueError("Unsupported optimizer")

    def _initialize_weights(self):
        """Initializes the weights and biases of the network."""
        for i in range(1, self.size):
            self.weights.append(np.random.randn(self.shape[i-1], self.shape[i]) * np.sqrt(2.0 / self.shape[i-1]))
            self.biases.append(np.zeros((1, self.shape[i])))

    def _initialize_batch_norm_params(self):
        """Initializes batch normalization parameters."""
        self.gamma = [np.ones((1, self.shape[i])) for i in range(1, self.size)]
        self.beta = [np.zeros((1, self.shape[i])) for i in range(1, self.size)]
        self.eps = 1e-8

    def _apply_activation(self, x, activation):
        """Applies the activation function."""
        if activation == 'sigmoid':
            return sigmoid(x)
        elif activation == 'relu':
            return relu(x)
        elif activation == 'tanh':
            return tanh(x)
        elif activation == 'leaky_relu':
            return leaky_relu(x)
        else:
            raise ValueError("Unsupported activation function")

    def _apply_activation_derivative(self, x, activation):
        """Applies the derivative of the activation function."""
        if activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif activation == 'relu':
            return relu_derivative(x)
        elif activation == 'tanh':
            return tanh_derivative(x)
        elif activation == 'leaky_relu':
            return leaky_relu_derivative(x)
        else:
            raise ValueError("Unsupported activation function")

    def _apply_batch_norm(self, x, gamma, beta):
        """Applies batch normalization."""
        mean = np.mean(x, axis=0, keepdims=True)
        variance = np.var(x, axis=0, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        return gamma * x_normalized + beta

    def _apply_dropout(self, x, rate):
        """Applies dropout."""
        if rate > 0:
            mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
            return x * mask
        return x

    def set_random_weights(self):
        """Sets random weights for the network."""
        for i in range(1, self.size):
            self.weights[i-1] = np.random.rand(self.shape[i-1], self.shape[i]) - 0.5
            self.biases[i-1] = np.random.rand(1, self.shape[i]) - 0.5

    def set_weights(self, weights, biases=None):
        for i in range(len(weights)):
            self.weights[i] = np.array(weights[i])
            if biases is not None:
                self.biases[i] = np.array(biases[i])

    def forward(self, inp):
        """Performs a forward pass through the network."""
        self.layers = [inp]
        for i in range(self.size - 1):
            z = np.dot(self.layers[-1], self.weights[i]) + self.biases[i]
            if self.use_batch_norm:
                z = self._apply_batch_norm(z, self.gamma[i], self.beta[i])
            a = self._apply_activation(z, self.activations[i])
            a = self._apply_dropout(a, self.dropout_rate)
            self.layers.append(a)
        return self.layers[-1]

    def backpropagate(self, inp, target):
        """Performs backpropagation to compute gradients."""
        m = target.shape[0]
        self.forward(inp)
        deltas = [None] * (self.size - 1)
        gradients_w = [None] * (self.size - 1)
        gradients_b = [None] * (self.size - 1)

        # Compute the delta for the output layer
        deltas[-1] = (self.layers[-1] - target) * self._apply_activation_derivative(self.layers[-1], self.activations[-1])

        # Compute the deltas for the hidden layers
        for i in reversed(range(self.size - 2)):
            deltas[i] = (np.dot(deltas[i + 1], self.weights[i + 1].T)) * self._apply_activation_derivative(self.layers[i + 1], self.activations[i])

        # Compute the gradients
        for i in range(self.size - 1):
            gradients_w[i] = np.dot(self.layers[i].T, deltas[i]) / m + (self.l2_lambda * self.weights[i])
            gradients_b[i] = np.sum(deltas[i], axis=0, keepdims=True) / m

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """Updates the weights using the gradients."""
        self.weights = self.optimizer.update(self.weights, gradients_w)
        for i in range(len(self.biases)):
            self.biases[i] -= self.optimizer.learning_rate * gradients_b[i]

    def train(self, X, y, epochs=100, batch_size=32, verbose=False):
        """Trains the neural network."""
        for epoch in range(epochs):
            # Shuffle the data
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                gradients_w, gradients_b = self.backpropagate(X_batch, y_batch)
                self.update_weights(gradients_w, gradients_b)

            if verbose:
                loss = np.mean(np.square(self.forward(X) - y))  # Mean Squared Error
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def get_deep_copy(self):
        """Creates a deep copy of the network."""
        n = NeuralNetwork(self.shape, self.activations, self.use_batch_norm, self.l2_lambda, self.dropout_rate, self.optimizer.__class__.__name__.lower(), self.optimizer.learning_rate)
        for i in range(self.size - 1):
            n.weights[i] = np.copy(self.weights[i])
            n.biases[i] = np.copy(self.biases[i])
            if self.use_batch_norm:
                n.gamma[i] = np.copy(self.gamma[i])
                n.beta[i] = np.copy(self.beta[i])
        return n

    def reproduce(self, mutation):
        """Creates a mutated copy of the network."""
        n = NeuralNetwork(self.shape, self.activations, self.use_batch_norm, self.l2_lambda, self.dropout_rate, self.optimizer.__class__.__name__.lower(), self.optimizer.learning_rate)
        for i in range(1, self.size):
            n.weights[i-1] = self.weights[i-1] + ((np.random.rand(self.shape[i-1], self.shape[i]) - 0.5) * mutation)
            n.biases[i-1] = self.biases[i-1] + ((np.random.rand(1, self.shape[i]) - 0.5) * mutation)
            if self.use_batch_norm:
                n.gamma[i-1] = self.gamma[i-1] + ((np.random.rand(1, self.shape[i]) - 0.5) * mutation)
                n.beta[i-1] = self.beta[i-1] + ((np.random.rand(1, self.shape[i]) - 0.5) * mutation)
        return n
