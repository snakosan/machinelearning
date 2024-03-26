import torch 
import torch.nn as nn
from tqdm import trange
from util import *
import matplotlib.pyplot as plt
X_train = fetch('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:].reshape((-1,28,28))
Y_train = fetch('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[8:]
X_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[0x10:].reshape((-1,28,28))
Y_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[8:]


class BobNet(nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(784,128)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.l2(self.act(self.l1(x)))
        return x

model = BobNet()

u = model(torch.tensor(X_train[0:10].reshape((-1, 28*28))).float())

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
BS = 32
num_epochs = 1000
losses, accuracies = [], []
def train(num_epochs, plot=False):
    for i in (t:= trange(num_epochs)):
        # get a vecotor of size BS of random values 
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        optim.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        acc = (cat == Y).float().mean()
        loss = loss_function(out,Y)
        loss.backward()
        optim.step()
        loss = loss.item()
        accuracy = acc.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description(f'loss: {loss:.2f} and accuracy : {accuracy:.2f}')
    if plot: 
        plt.plot(losses)
        plt.plot(accuracies)
        plt.show()
    return losses, accuracies

train(10000)


x_test = torch.Tensor(X_test.reshape((-1, 28*28))).float()
y_test = model(x_test)
y_test = (torch.argmax(y_test,1)).numpy()
ac = (Y_test == y_test).mean()
print(ac)



#using numpy, first to evaluate the model 

l1 = np.zeros((784, 128), dtype=np.float32)
l2 = np.zeros((128,10), dtype=np.float32)
l1[:] = model.l1.weight.detach().numpy().transpose()
l2[:] = model.l2.weight.detach().numpy().transpose()
print(l1.dtype, l2.dtype, id(l1), id(l2))
def forward(X):
    X = X.dot(l1)
    X = np.maximum(X,0)
    X = X.dot(l2)
    return X

y_pred = np.argmax(forward(X_test.reshape((-1, 28*28))), axis=1)
acc_np = (y_pred == Y_test).mean()
print (acc_np)
