import numpy as np

def sigmoid_deriv(a):
	return a * (1 - a)

def softmax_deriv(a):
	return a

class BackpropHiddenOutput:
	def __init__(self, activation='sigmoid'):
		self.activation = activation

	def compute_gradients(self, H, y_true, y_pred, z, W):
		m = y_true.shape[0]
		if self.activation == 'sigmoid':
			dz = (y_pred - y_true) * sigmoid_deriv(y_pred)
		else:  # softmax + cross-entropy
			dz = (y_pred - y_true)
		dW = H.T @ dz / m
		db = np.sum(dz, axis=0, keepdims=True) / m
		return dW, db, dz