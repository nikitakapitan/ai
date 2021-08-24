import matplotlib.pyplot as plt
import numpy as np
from ai.dls.seg.addi import BATCH_SIZE

def plot_seg(X_val, answer):
    plt.figure(figsize=(18, 6))
    for k in range(5):
        n = min(BATCH_SIZE, 6)
        plt.subplot(2, n, k + 1)
        plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
        plt.title('Real')
        plt.axis('off')

        plt.subplot(2, n, k + 1 + n)
        print(answer.shape)
        print(answer[k].shape)
        print(answer[k, 0].shape)
        plt.imshow(answer[k, 0], cmap='gray')
        plt.title('Output')
        plt.axis('off')
    return plt