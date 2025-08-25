import numpy as np
import matplotlib.pyplot as plt
from MLP.forward_input_hidden import ForwardInputHidden
from MLP.forward_hidden_output import ForwardHiddenOutput
from MLP.weight_update import update_weights

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_dim, hidden_dim, output_dim = 2, 2, 1
fwd1 = ForwardInputHidden(input_dim, hidden_dim, activation='sigmoid')
fwd2 = ForwardHiddenOutput(hidden_dim, output_dim, activation='sigmoid', loss='mse')
lr = 0.1
epochs = 2000
weights1, weights2 = [], []

for epoch in range(epochs):
	a1, z1 = fwd1.forward(X)
	y_pred, z2 = fwd2.forward(a1)
	dz2 = (y_pred - y) * y_pred * (1 - y_pred)
	dW2 = a1.T @ dz2 / X.shape[0]
	db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]
	dz1 = (dz2 @ fwd2.W.T) * a1 * (1 - a1)
	dW1 = X.T @ dz1 / X.shape[0]
	db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]
	fwd2.W, fwd2.b = update_weights(fwd2.W, dW2, fwd2.b, db2, lr)
	fwd1.W, fwd1.b = update_weights(fwd1.W, dW1, fwd1.b, db1, lr)
	if epoch % 100 == 0:
		weights1.append(fwd1.W.copy())
		weights2.append(fwd2.W.copy())

weights1 = np.array(weights1)
weights2 = np.array(weights2)

plt.figure(figsize=(10,5))
for i in range(weights1.shape[2]):
	plt.plot(weights1[:,0,i], label=f'Input-Hidden W[0,{i}]')
	plt.plot(weights1[:,1,i], label=f'Input-Hidden W[1,{i}]')
for i in range(weights2.shape[1]):
	plt.plot(weights2[:,0,i], label=f'Hidden-Output W[0,{i}]')
plt.legend()
plt.title('Weight Updates Over Epochs')
plt.xlabel('Epoch (x100)')
plt.ylabel('Weight Value')
plt.show()