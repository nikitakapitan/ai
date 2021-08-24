from skimage.io import imread
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 2
IMG_SIZE = 256

images = []
lesions = []
root = 'PH2Dataset'


for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
    if root.endswith('_Dermoscopic_Image'):
        images.append(imread(os.path.join(root, files[0])))
    if root.endswith('_lesion'):
        lesions.append(imread(os.path.join(root, files[0])))


size = (IMG_SIZE, IMG_SIZE)
X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

X = np.array(X, np.float32)
Y = np.array(Y, np.float32)


# plt.figure(figsize=(18, 6))
# for i in range(6):
#     plt.subplot(2, 6, i+1)
#     plt.axis("off")
#     plt.imshow(X[i])
#
#     plt.subplot(2, 6, i+7)
#     plt.axis("off")
#     plt.imshow(Y[i])
# plt.show()

ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [100, 150])

batch_size = BATCH_SIZE
data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
                     batch_size=batch_size, shuffle=True, pin_memory=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                      batch_size=batch_size, shuffle=True, pin_memory=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
                     batch_size=batch_size, shuffle=True, pin_memory=True)
