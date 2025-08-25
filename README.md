# MLP-XOR: Multi-Layer Perceptron from Scratch

This project implements and experiments with a simple Multi-Layer Perceptron (MLP) to solve the XOR problem, covering both the core algorithm and experimental analysis. Below are the activities performed in this project:

## Part A – Implementation (Core MLP)

1. **Forward Pass (Input → Hidden Layer)**
   - Matrix multiplication from input to hidden layer
   - Configurable activation function (Sigmoid or ReLU)

2. **Forward Pass (Hidden → Output Layer)**
   - Matrix multiplication from hidden to output layer
   - Configurable output activation (Sigmoid or Softmax)
   - Choice of loss function: Mean Squared Error (MSE) or Cross-Entropy

3. **Backpropagation (Hidden → Output)**
   - Derivation and implementation of gradients for output layer weights and biases

4. **Backpropagation (Input → Hidden)**
   - Derivation and implementation of gradients for hidden layer weights and biases

5. **Weight Update Rules**
   - Gradient descent weight updates
   - Configurable learning rate
   - Training loop over multiple epochs

## Part B – Verification & Experiments

6. **Numerical Gradient Check**
   - Finite difference approximation of gradients
   - Comparison with analytical backpropagation gradients

7. **XOR Dataset Setup + Training**
   - Encoding the XOR truth table
   - Training the MLP on XOR
   - Demonstrating loss decrease over epochs

8. **Decision Boundary Visualization**
   - Plotting decision regions for XOR before and after training
   - Showing how the MLP solves the non-linear XOR problem

9. **Weight Update Visualization**
   - Tracking and plotting weight updates over epochs
   - Visualizing the learning progression

---

## How to Run

1. Clone the repository and install requirements (see `requirements.txt`).
2. Run the main script to train and visualize the MLP on XOR.
3. Check the code and plots for each activity above.

---

**Author:** [Vaibhav Jain](https://github.com/v-aibha-v-jain)

**Date:** August 2025
