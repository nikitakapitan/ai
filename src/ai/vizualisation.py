import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (8, 6.)})
sns.set_style("whitegrid")

def target(training, test, valid, ax):
    training[0] = (test[0] + valid[0])/2 # Pretty plot only. after 1st epoch we are already better on train.
    ax.set_title('Target function performance training vs validation data')
    ax.plot(training, label='training')
    ax.plot(test, label='test')
    ax.plot(valid, label='validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cost')
    ax.legend()
    
def misclassification(training, test, valid, ax):
    training[0] = (test[0] + valid[0])/2  # Pretty plot only. after 1st epoch we are already better on train.
    ax.set_title('Classification accuracy: #(predict is correct)/#all')
    ax.plot(training, label='training')
    ax.plot(test, label='test')
    ax.plot(valid, label='validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    
    

    
    
