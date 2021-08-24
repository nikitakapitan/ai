import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from time import time

from ai.dls.seg.vizual import plot_seg


class MarkusSegNet(nn.Module):
    def __init__(self):
        """
        After initializng in iPython, write markus.to(device)
        """
        super(MarkusSegNet, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        for param in self.parameters():
            param.requires_grad = False

        self.upsampling = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.classifier = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, X):        # [batch, C(3),   H(256), W(256)]
        x = self.features(X)     # [batch, C(512), H(8)  , W(8)  ]
        x = self.upsampling(x)   # [batch, C(512), H(256), W(256)]
        x = self.classifier(x)   # [batch, C(1)  , H(256), W(256)]
        x = F.sigmoid(x)         # [probabilities: output on [0,1] (noo need softmax as binary seg)
        return x                 # [batch, C(1)  , H(256), W(256)]

    def fit(model, opt, loss_fn, epochs, data_tr : DataLoader, data_val : DataLoader, device='cpu'):
        """
        opt: torch.optim, this optimazed should be externaly charged with model.parameters()
        loss_fn: function, any function. Optimizer knowns nothing about loss.
        The infromation about loss function will be incorporated into model.parameters() after loss.backward()
        epochs: int
        data_tr. Normally on CPU, will be put on available device.
        """
        X_val, Y_val = next(iter(data_val))
        if epochs == 0:
            answer = Y_val

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            sum_loss = 0

            # LEARNING
            model.train()
            for X_batch, Y_batch in data_tr:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                opt.zero_grad()                     # clear model.parameters gradient

                pred = model.forward(X_batch)       # forward -> [batch, 1, 256, 256]
                loss = loss_fn(pred, Y_batch)
                sum_loss += loss                    # user info

                loss.backward()                     # compute gradient for each model parameter
                opt.step()                          # apply gradient step
            print(f'Train Avg epoch({epoch}) loss = ', sum_loss/len(data_tr))

            # BENCHMARKING
            model.eval()
            X_val = X_val.to(device)
            answer = model.forward(X_val)           # [batch, C(1)  , H(256), W(256)]
            X_val, answer = X_val.detach().cpu(), answer.detach().cpu()
            tloss = loss_fn(answer, Y_val)
            print(f'Test Avg epoch {epoch} loss = ', tloss/len(X_val))

        model.plot = plot_seg(X_val, answer)
