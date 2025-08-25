import numpy as np

def sigmoid_deriv(a):
	return a * (1 - a)

def relu_deriv(z):
	return (z > 0).astype(float)

class BackpropInputHidden:
	def __init__(self, activation='sigmoid'):
		self.activation = activation

	def compute_gradients(self, X, dz_next, W_next, z, a):
		m = X.shape[0]
		if self.activation == 'sigmoid':
			dz = (dz_next @ W_next.T) * sigmoid_deriv(a)
		else:
			dz = (dz_next @ W_next.T) * relu_deriv(z)
		dW = X.T @ dz / m
		db = np.sum(dz, axis=0, keepdims=True) / m
		return dW, db, dz