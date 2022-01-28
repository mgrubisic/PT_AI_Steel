import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler
import numpy as np

def readEncoding():
    with open("SP/Encoding.txt") as tF:
        lines = [line.strip().split('\t') for line in tF]
    return lines


class Model(nn.Module):
    def __init__(self, n_input, n1, n2, n_output):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, n_output),
        )

    def forward(self, x):
        if x.shape == torch.Size([3]):
            x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        output = F.log_softmax(logits, dim=1)
        return output

    def set(self, *, optimizer=None, loss=None, scheduler=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if scheduler is not None:
            self.scheduler = scheduler

    def train_loop(self, X, y, epochs):

        for epoch in range (1, epochs+1):
            pred = self.forward(X)
            lossObj = self.loss(pred, y.long())

            # Backpropagation
            self.optimizer.zero_grad()
            lossObj.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                lossObj = lossObj.item()
                info = self.scheduler.state_dict()
                info2 = info['_last_lr']
                print(f"LR: {info2[0]:>7f}, loss: {lossObj:>7f}, epoch: {epoch}")
            if epoch % 1000 == 0:
                correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                correct /= len(X)
                print(f"Correct: {(100*correct):>0.1f}%")

            if epoch % 50 == 0:
                self.scheduler.step()


    def test_loop(self, X_test, y_test):
        size = len(X_test)
        test_loss, correct = 0, 0
        with torch.no_grad():
            pred = self.forwardPred(X_test)
            test_loss += self.loss(pred, y_test).item()
            correct += (pred.argmax(1) == y_test).type(torch.float).sum().item()
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss



def askModel(p_load, v_load, span):
    with open("bestModel.model", "rb") as f:
        model = pickle.load(f)
    print("Model load: success")
    x = torch.tensor([[p_load, v_load, span]])
    x = x.float()
    x = x.to("cuda:0")

    with torch.no_grad():
        pred = model(x)
        pred = pred.cpu()
        pred_profile = np.argmax(pred)

        labels = readEncoding()

        print(labels[pred_profile][2])






askModel(4, 5, 6)