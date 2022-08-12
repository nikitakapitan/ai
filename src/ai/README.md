Hi, this folder was my very first step into AI world. 

I simply started from empty.py files and wrote on my own the fully connected neural network and learning algorithm.

*Purpose*: knowledge 

*Application*: MNIST classification

*Interactive example*: ai/notebooks/gold standard/recon.ipynb

<h2>Structure</h2>

<h3>network.py </h3> 
Neural network class definition with:

- def __ init __ : network initialization with activation func, loss func, l1 & l2 regulalization
- def predict: (aka forward pass) : return last layer activation
- def fit (aka train) : shuffle, batch split, **update_batch**, compute metrics (ex. accuracy) and debug traces
  - def update_batch: update weights given the pre-computed gradient from **backpropa**
    - def backpropa : compute analytical gradient through the whole network

<h3>activation_function.py </h3>
abstract class for general activation function.
  - you can define your own activation function to be used for training

available examples: Linear, Sigmoid

<h3>target_functions.py </h3>(aka Loss)
abstract class for general Loss function

 - you can define your own loss function to be used for training 
 
 available examples: QuadraticLoss, CrossEntropyLoss

<h3>gradient.py</h3> 
additional file with analytical and numerical gradiend analysis

<h3>regnetwork.py</h3> 
Inherited network class with l1 and l2 regularization

<h3>neuron.py</h3> (depricated, not used) - class definition of a single neuron, aka Perceptron.
