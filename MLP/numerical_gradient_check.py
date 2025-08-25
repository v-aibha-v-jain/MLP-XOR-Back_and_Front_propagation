import numpy as np

def numerical_gradient(f, params, eps=1e-5):
	grad = np.zeros_like(params)
	for i in range(params.size):
		orig = params.flat[i]
		params.flat[i] = orig + eps
		fx1 = f(params)
		params.flat[i] = orig - eps
		fx2 = f(params)
		grad.flat[i] = (fx1 - fx2) / (2 * eps)
		params.flat[i] = orig
	return grad