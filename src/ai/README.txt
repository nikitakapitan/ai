Hi, this folder was my very first step into AI world.

Purpose: knowledge 
Application: MNIST classification
Interactive example: ai/notebooks/gold standard/recon.ipynb

I simply started from empty.py files and wrote on my own the fully connected neural network.

***Structure***:

network.py [Main file] - neural network class definition with:
- def __init__ : network initialization with activation func, loss func, l1 & l2 regulalization
- def predict: (aka forward pass) : return last layer activation
- def fit (aka train) : shuffle, batch split, compute metrics (ex. accuracy) and debug traces
  - def update_batch: update weights given the pre-computed gradient
    - def backpropa : compute analytical gradient through the whole network

activation_function.py - abstract class for general activation function.
  - you can define your own activation function to be used for training

available examples: Linear, Sigmoid

target_functions.py (aka Loss) - abstract class for general Loss function
 - you can define your own loss function to be used for training 
 
 available examples: QuadraticLoss, CrossEntropyLoss

gradient.py - additional file with analytical and numerical gradiend analysis

regnetwork.py - Inherited network class with regularization

neuron.py (depricated, not used) - class definition of a single neuron, aka Perceptron.
