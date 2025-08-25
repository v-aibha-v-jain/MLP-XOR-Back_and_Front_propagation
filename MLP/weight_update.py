import numpy as np

def update_weights(W, dW, b, db, lr):
	W -= lr * dW
	b -= lr * db
	return W, b

def train_loop(model, X, y, epochs=1000, lr=0.1):
	losses = []
	for epoch in range(epochs):
		y_pred, _ = model.forward(X)
		loss = model.compute_loss(y, y_pred)
		losses.append(loss)
		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Loss: {loss}")
	return losses