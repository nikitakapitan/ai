import matplotlib.pyplot as plt
import numpy as np

def plot_seg(X_val, answer):

    for k in range(5):
        plt.subplot(2, 6, k + 1)
        plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
        plt.title('Real')
        plt.axis('off')

        plt.subplot(2, 6, k + 7)
        plt.imshow(answer[k], cmap='gray')
        plt.title('Output')
        plt.axis('off')
    plt.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))
    model.plot = plt