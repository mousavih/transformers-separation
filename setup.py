import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_input(n, N, d, simple=False):
    X = torch.randn(n, N, d).to(device)
    if not simple:
        t = torch.randint(N, size=(n, N)).to(device)
    else:
        t = torch.randint(N, size=(n,1)).repeat(1, N).to(device)

    return X, t

def embed_input(X, t, E):
    n = X.shape[0]
    return torch.cat([X, E.repeat(n, 1, 1), E[t]], dim=-1)

def ground_truth(X, t, u):
    d = X.shape[-1]
    X_select = torch.gather(X, 1, t.unsqueeze(-1).repeat(1, 1, d))
    
    return X_select @ u

def test_MSE(model, testloader):
    with torch.no_grad():
        test_loss = 0
        for XE_batch, y_batch in testloader:
            test_loss += (model(XE_batch) - y_batch).pow(2).mean(dim=1).sum().item() / len(testloader.dataset)
    return test_loss


class SimpleTransformer(nn.Module):
    def __init__(self, d, de, m):
        super().__init__()
        embed_dim = d + 2 * de
        
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )
        
    def forward(self, x):
        return self.mlp(self.attn(x, x, x, need_weights=False)[0])
    
    
class FFN(nn.Module):
    def __init__(self, embed_dim, N, hidden_dims):
        super().__init__()
        self.l1 = nn.Linear(embed_dim * N, hidden_dims[0])
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])
        self.l2 = nn.Linear(hidden_dims[-1], N)
        
        self.embed_dim = embed_dim
        self.N = N
        
    def forward(self, x):
        x = torch.relu(self.l1(x.view(-1, self.embed_dim * self.N)))
        for l in self.layers:
            x = torch.relu(l(x))
        
        return self.l2(x).unsqueeze(-1)
    
    
class SimpleBRNN(nn.Module):
    def __init__(self, d, de, h_size, bidirectional):
        super().__init__()
        self.rnn = nn.RNN(d + 2 * de, h_size, batch_first=True, bidirectional=bidirectional, nonlinearity='relu')
        if bidirectional:
            self.linear = nn.Linear(2 * h_size, 1)
        else:
            self.linear = nn.Linear(h_size, 1)
        
    def forward(self, x):
        x = self.rnn(x)[0]
        return self.linear(x)
    
    
class BLSTM(nn.Module):
    def __init__(self, d, de, h_size, bidirectional):
        super().__init__()
        self.rnn = nn.LSTM(d + 2 * de, h_size, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(2 * h_size, 1)
        else:
            self.linear = nn.Linear(h_size, 1)
    
    def forward(self, x):
        x = self.rnn(x)[0]
        return self.linear(x)
    
class BGRU(nn.Module):
    def __init__(self, d, de, h_size, bidirectional):
        super().__init__()
        self.rnn = nn.GRU(d + 2 * de, h_size, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(2 * h_size, 1)
        else:
            self.linear = nn.Linear(h_size, 1)
    
    def forward(self, x):
        x = self.rnn(x)[0]
        return self.linear(x)


def train_model_qstr(model, N, E, trainloader, XE_test, y_test, epochs=20, lr=0.001, return_train_loss=False, verbose=False, weight_decay=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []

    for e in range(epochs):
        running_loss = 0.
        for i, batch in enumerate(trainloader):
            X_batch, t_batch, y_batch = batch
            XE_batch = embed_input(X_batch, t_batch, E)
            opt.zero_grad()
            loss = (model(XE_batch) - y_batch).pow(2).mean()
            loss.backward()
            opt.step()

            if verbose and i % (len(trainloader) // 5) == 0:
                print(f'N: {N}, Epoch: {e}, i: {i}, loss: {loss.item()}')
            
            running_loss += loss.item() / len(trainloader)
        
        train_loss.append(running_loss)

        if running_loss < 0.05:
            break

    with torch.no_grad():
        loss = (model(XE_test) - y_test).pow(2).mean().item()

    print(f'N: {N}, test loss: {loss}', flush=True)
    print()
    if return_train_loss:
        return loss, train_loss
    else:
        return loss
    

def find_sample_complexity(N, d, de, model, u, E, testloader, max_n_train=1000000, test_batch_size=1000, loss_thr=0.5, lr=0.001, batch_size=64, simple=False, print_every=1000, check_every=10, weight_decay=0.01):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = 0

    while n_train < max_n_train:
        X_batch, t_batch = generate_input(batch_size, N, d, simple)
        y_batch = ground_truth(X_batch, t_batch, u)

        XE_batch = embed_input(X_batch, t_batch, E)
        opt.zero_grad()
        loss = (model(XE_batch) - y_batch).pow(2).mean()
        loss.backward()
        opt.step()

        n_train += batch_size
        
        if (n_train // batch_size) % check_every == 0:
            with torch.no_grad():
                test_loss = test_MSE(model, testloader)
            if test_loss < loss_thr:
                return n_train

        if (n_train // batch_size) % print_every == 0:
            print(f'n_train: {n_train}, loss: {loss.item()}', flush=True)
            
            print(f'n_train: {n_train}, test loss: {test_MSE(model, testloader)}', flush=True)
        
    raise Exception('Did not converge')