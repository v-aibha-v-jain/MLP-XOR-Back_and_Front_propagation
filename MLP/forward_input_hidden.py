import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

class ForwardInputHidden:
	def __init__(self, input_dim, hidden_dim, activation='sigmoid'):
		self.W = np.random.randn(input_dim, hidden_dim) * 0.1
		self.b = np.zeros((1, hidden_dim))
		self.activation = sigmoid if activation == 'sigmoid' else relu

	def forward(self, X):
		z = X @ self.W + self.b
		a = self.activation(z)
		return a, z