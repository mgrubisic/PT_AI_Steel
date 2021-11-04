import pickle
import matplotlib.pyplot as plt
import time

from skopt import BayesSearchCV, gp_minimize
from skopt.plots import plot_convergence, plot_objective_2D, plot_objective
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler


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
                #print(f"LR: {info2[0]:>7f}, loss: {lossObj:>7f}, epoch: {epoch}")
            if epoch % 1000 == 0:
                correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                correct /= len(X)
                print(f"Corret: {(100*correct):>0.1f}%")

            if epoch % 50 == 0:
                self.scheduler.step()


    def test_loop(self, X_test, y_test):
        size = len(X_test)
        test_loss, correct = 0, 0
        with torch.no_grad():
            pred = self.forward(X_test)
            test_loss += self.loss(pred, y_test).item()
            correct += (pred.argmax(1) == y_test).type(torch.float).sum().item()
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss

# Read in training data
def readTrainingData():

    with open("SP/trainingData_Tensor_pickle.pk", 'rb') as fi1:
        X_training = pickle.load(fi1)
        X_training = X_training.float()
    with open("SP/trainingData_Y_Tensor_pickle.pk", 'rb') as fi2:
        y_training = pickle.load(fi2)
        y_training = y_training.long()
    return X_training, y_training

# read test/validation data
def readTestData():

    with open("SP/testData_Tensor_pickle.pk", 'rb') as fti1:
        X_test = pickle.load(fti1)
        X_test = X_test.float()
    with open("SP/testData_Y_Tensor_pickle.pk", 'rb') as fti2:
        y_test = pickle.load(fti2)
        y_test = y_test.long()
    return X_test, y_test

# dictionary to translate number to a steel profile (
def readEncoding():
    with open("SP/Encoding.txt") as tF:
        lines = [line.strip().split('\t') for line in tF]
    return lines

#-------------------------------------------------------------
# FLAGS - change boolean value to control program
to_GPU = True
TRAIN_GPU_MODEL = False
TRAIN_POST_BAY = True
DO_BAY_OPT = False
# -------------------------------------------------------------
t = time.time()

if TRAIN_GPU_MODEL:
    if to_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu:0")

    dim_neurons1 = Integer(low=15, high=65, name="neurons1")
    dim_neurons2 = Integer(low=65, high=110, name="neurons2")
    dim_LR = Real(low=9e-3, high=0.1, name="LR")
    dim_decay_LR = Real(low=1e-5, high=1e-1, prior="log-uniform", name="decay_LR")
    dimensions = [dim_neurons1, dim_neurons2, dim_LR, dim_decay_LR]

    def create_model(neurons1, neurons2, LR, decay_LR):
        model = Model(3, neurons1, neurons2, 148)
        loss_function = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=1-decay_LR)
        model.set(optimizer=opt, loss=loss_function, scheduler=scheduler)
        model = model.to(device)

        return model

    @use_named_args(dimensions=dimensions)
    def fitness(neurons1, neurons2, LR, decay_LR):
        global X_training, y_training, X_test, y_test, itera, device
        # Print the hyper-parameters

        print('Number of neurons1     : {0:.2e}'.format(neurons1))
        print('Number of neurons2     : {0:.2e}'.format(neurons2))
        print('LR                   : {0:.2e}'.format(LR))
        print('decay                : {0:.2e}'.format(decay_LR))

        # Count and print to console to keep track of fitness progress
        itera += 1
        print(f' \t \t \t Iteration number: {itera}')

        # Create the neural network
        model = create_model(neurons1, neurons2, LR, decay_LR)
        model.train_loop(X_training, y_training, 11000)

        return model.test_loop(X_test, y_test)

    itera = 0

    if DO_BAY_OPT:
        # Default parameters
        X_training, y_training = readTrainingData()
        X_test, y_test = readTestData()
        X_training, y_training = X_training.to(device), y_training.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        default_parameters = [25, 100, 0.028, 0.0001]

        search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func="EI",
                                    n_calls=150,
                                    x0=default_parameters)
        with open("pytorchOpt1.pk", 'wb') as f:
            pickle.dump(search_result, f)
    else:
        with open("pytorchOpt1.pk", "rb") as f:
            search_result = pickle.load(f)

        print('Search_result.x:')
        print(f'Neurons1: {search_result.x[0]},' +
              f'Neurons2: {search_result.x[1]},' +
              f'LR L1: {search_result.x[2]:.3e},' +
              f'Decay L1: {search_result.x[3]:.3e}')

        print("sorted(zip(search_result.func_vals, search_result.x_iters))")
        print(sorted(zip(search_result.func_vals, search_result.x_iters)))

        fig = plot_objective_2D(result=search_result,
                                dimension_identifier1='LR',
                                dimension_identifier2='decay_LR',
                                levels=100)
        plt.savefig("lay2OPT.png", dpi=400)
        plt.show()

        # Create a list for plotting
        dim_names = ['neurons1', 'neurons2','LR', 'decay_LR']
        # fig, ax = plot_objective(result=search_result, dimensions=dim_names)
        _ = plot_objective(result=search_result, dimensions=dim_names)
        plt.savefig("all_dimen_layer1.png", dpi=400)
        plt.show()

    """   
    my_nn = Model(3, 19, 101, 148)
    my_nn = my_nn.to(device)

    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(my_nn.parameters(), lr=0.06)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=1-decay_LR)
    my_nn.set(opt, loss=loss_function)

    train_loop(X_training, y_training, X_test, y_test, my_nn, loss_function, opt, 10000, scheduler)
    """

if TRAIN_POST_BAY:
    if to_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu:0")
    X_training, y_training = readTrainingData()
    X_test, y_test = readTestData()
    X_training, y_training = X_training.to(device), y_training.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    def fitness(neurons1, neurons2, LR, decay_LR):
        global X_training, y_training, X_test, y_test, itera, device
        # Print the hyper-parameters

        print('Number of neurons1     : {0:.2e}'.format(neurons1))
        print('Number of neurons2     : {0:.2e}'.format(neurons2))
        print('LR                   : {0:.2e}'.format(LR))
        print('decay                : {0:.2e}'.format(decay_LR))

        # Count and print to console to keep track of fitness progress
        itera += 1
        print(f' \t \t \t Iteration number: {itera}')

        # Create the neural network
        model = create_model(neurons1, neurons2, LR, decay_LR)
        model.train_loop(X_training, y_training, 20000)

        return model.test_loop(X_test, y_test)

    with open("pytorchOpt1.pk", "rb") as f:
        search_result = pickle.load(f)

    model = Model(3, search_result.x[0], search_result.x[1], 148)
    loss_function = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=search_result.x[2])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=1 - search_result.x[3])
    model.set(optimizer=opt, loss=loss_function, scheduler=scheduler)
    model = model.to(device)

    model.train_loop(X_training, y_training, 13000)
    res = model.test_loop(X_test, y_test)
    print(f"Loss: {res}")


print(time.time() - t)

