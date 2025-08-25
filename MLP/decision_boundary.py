import numpy as np
import matplotlib.pyplot as plt
from forward_input_hidden import ForwardInputHidden
from forward_hidden_output import ForwardHiddenOutput

def plot_decision_boundary(fwd1, fwd2, X, y, title):
	h = 0.01
	x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
	y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	grid = np.c_[xx.ravel(), yy.ravel()]
	a1, _ = fwd1.forward(grid)
	y_pred, _ = fwd2.forward(a1)
	Z = (y_pred > 0.5).astype(int).reshape(xx.shape)
	plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
	plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.coolwarm)
	plt.title(title)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()