# xor problem
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import random
import itertools
import torch.nn.init as init

class xorProblem(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack= nn.Sequential(
            nn.Linear(3,4),
            nn.LeakyReLU(0.01),
            nn.Linear(4,4),
            nn.LeakyReLU(0.01),
            nn.Linear(4,4),
            nn.LeakyReLU(0.01),
            nn.Linear(4,2),
            nn.LeakyReLU(0.01),
            nn.Linear(2,1),
            nn.Sigmoid()
        )
        init.kaiming_uniform_(self.stack[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.stack[2].weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.stack[4].weight, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.stack[6].weight, mode='fan_in', nonlinearity='leaky_relu')
        init.xavier_uniform_(self.stack[8].weight)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits
    
model=xorProblem().to("cpu")
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()
datain=[]
dataout=[]
for i,j,k in itertools.product(range(0,2),repeat=3):
    datain.append([i,j,k])
    dataout.append([i^j^k])
for i in range(10001):
    inp=[float(random.randint(0,1)),float(random.randint(0,1))]
    out=[float(bool(inp[0])^bool(inp[1]))]
    #outp=torch.tensor([out], dtype=torch.float32)
    #X = torch.tensor([inp], dtype=torch.float32)
    indices = random.sample(range(len(datain)), 4)

    # Sample the same indices from both lists
    datainb = [datain[i] for i in indices]
    dataoutb = [dataout[i] for i in indices]
    outp = torch.tensor(dataoutb, dtype=torch.float32)
    X=torch.tensor(datainb, dtype=torch.float32)
    res = model(X)
    model.train()
    loss = criterion(res, outp)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i%1000==0):
        for j in range(len(datainb)):
            print(f"Round {i}: Input: {X[j].tolist()}, Expected output: {outp[j].tolist()}, got: {res[j].tolist()}, loss: {loss.tolist()}")
print("Network trained! Enter data:")
while True:
    inp=input("numbers: ").split(",")
    inp=[float(inp[i]) for i in range(3)]
    X=torch.tensor([inp], dtype=torch.float32)
    res = model(X)
    print(f"Output: {res[0].tolist()[0]}")