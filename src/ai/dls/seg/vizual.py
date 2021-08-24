import matplotlib.pyplot as plt
import numpy as np
from ai.dls.seg.addi import BATCH_SIZE

def plot_seg(X_val, answer, ground):
    plt.figure(figsize=(18, 6))
    for k in range(5):
        n = min(BATCH_SIZE, 6)
        plt.subplot(4, n, k + 1)
        plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
        plt.title('Real')
        plt.axis('off')

        plt.subplot(4, n, k + 1 + n)
        plt.imshow(answer[k, 0], cmap='gray')
        plt.title('Output')
        plt.axis('off')

        quantile = 0.75
        seg = (answer[k,0] > np.quantile(answer[k,0].numpy(), quantile)).int()
        plt.subplot(4, n, k + 1 + 2 * n)
        plt.imshow(seg, cmap='gray')
        plt.title(f'Binary seg, quantile={quantile}')
        plt.axis('off')

        plt.subplot(4, n, k + 1 + 3 * n)
        plt.imshow(ground, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
    return plt

def plot_learning(epochs,train_avgloss, test_avgloss):
    plt.plot(range(epochs), train_avgloss, label='Train')
    plt.plot(range(epochs), test_avgloss, label='Test')
    plt.title('Learning')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.show()