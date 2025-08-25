import numpy as np
from forward_input_hidden import ForwardInputHidden
from forward_hidden_output import ForwardHiddenOutput
from weight_update import update_weights

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_dim, hidden_dim, output_dim = 2, 2, 1
fwd1 = ForwardInputHidden(input_dim, hidden_dim, activation='sigmoid')
fwd2 = ForwardHiddenOutput(hidden_dim, output_dim, activation='sigmoid', loss='mse')
lr = 0.1
epochs = 10000
losses = []

for epoch in range(epochs):
	a1, z1 = fwd1.forward(X)
	y_pred, z2 = fwd2.forward(a1)
	loss = fwd2.compute_loss(y, y_pred)
	losses.append(loss)
	dz2 = (y_pred - y) * y_pred * (1 - y_pred)
	dW2 = a1.T @ dz2 / X.shape[0]
	db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]
	dz1 = (dz2 @ fwd2.W.T) * a1 * (1 - a1)
	dW1 = X.T @ dz1 / X.shape[0]
	db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]
	fwd2.W, fwd2.b = update_weights(fwd2.W, dW2, fwd2.b, db2, lr)
	fwd1.W, fwd1.b = update_weights(fwd1.W, dW1, fwd1.b, db1, lr)
	if epoch % 1000 == 0:
		print(f"Epoch {epoch}, Loss: {loss}")