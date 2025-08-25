import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
	e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return e_x / np.sum(e_x, axis=1, keepdims=True)

def mse_loss(y_true, y_pred):
	return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
	eps = 1e-12
	y_pred = np.clip(y_pred, eps, 1 - eps)
	return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class ForwardHiddenOutput:
	def __init__(self, hidden_dim, output_dim, activation='sigmoid', loss='mse'):
		self.W = np.random.randn(hidden_dim, output_dim) * 0.1
		self.b = np.zeros((1, output_dim))
		self.activation = sigmoid if activation == 'sigmoid' else softmax
		self.loss = mse_loss if loss == 'mse' else cross_entropy_loss

	def forward(self, H):
		z = H @ self.W + self.b
		a = self.activation(z)
		return a, z

	def compute_loss(self, y_true, y_pred):
		return self.loss(y_true, y_pred)