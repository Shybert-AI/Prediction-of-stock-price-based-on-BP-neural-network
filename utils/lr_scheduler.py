# -*- coding:utf-8 -*-

#1. torch.optim.lr_scheduler.StepLR  等间隔调整学习率 
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
step_schedule = optim.lr_scheduler.StepLR(step_size=20, gamma=0.9, optimizer=optimizer)
step_lr_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_schedule.step()
    step_lr_list.append(step_schedule.get_last_lr()[0])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(step_lr_list)), step_lr_list, label="step_lr")
plt.legend()
plt.savefig("StepLR.png")
plt.show()


#2.torch.optim.lr_scheduler.MultiStepLR  按需调整学习率 MultiStepLR
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
multi_schedule = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[120,180])
multi_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    multi_schedule.step()
    multi_list.append(multi_schedule.get_last_lr()[0])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(multi_list)), multi_list, label="step_lr")
plt.legend()
plt.savefig("MultiStepLR.png")
plt.show()

#3、指数衰减调整学习率 ExponentialLR
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
exponent_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
multi_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    exponent_schedule.step()
    multi_list.append(exponent_schedule.get_last_lr()[0])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(multi_list)), multi_list, label="step_lr")
plt.legend()
plt.savefig("ExponentialLR.png")
plt.show()

#4、余弦退火调整学习率 CosineAnnealingLR
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=0.0004)
multi_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cosine_schedule.step()
    multi_list.append(cosine_schedule.get_last_lr()[0])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(multi_list)), multi_list, label="cosine_lr")
plt.legend()
plt.savefig("CosineAnnealingLR.png")
plt.show()

#5、自适应调整学习率 ReduceLROnPlateau
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out


model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                       verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8)
multi_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    reduce_schedule.step(loss)
    multi_list.append(optimizer.param_groups[0]["lr"])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(multi_list)), multi_list, label="reduce_lr")
plt.legend()
plt.savefig("ReduceLROnPlateau.png")
plt.show()

#6、自定义调整学习率 LambdaLR
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out

lambda1 = lambda epoch: epoch//20
lambda2 = lambda epoch: 0.95**epoch
model = LinearRegression()
print(model.linear1)
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = optim.SGD(
    [{"params": model.linear1.parameters(), "lr": 0.01},
     {"params": model.linear2.parameters()}],
    lr=0.02)
lambda_schedule = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=[lambda1,lambda2])
lambda1_list = []
lambda2_list = []
loss_list = []
# 开始训练
num_epochs = 240
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lambda_schedule.step()
    lambda1_list.append(optimizer.param_groups[0]["lr"])
    lambda2_list.append(optimizer.param_groups[1]["lr"])
    loss_list.append(loss.item())
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.legend()
plt.subplot(122)
plt.plot(range(len(lambda1_list)),lambda1_list,label="lambda1_lr")
plt.plot(range(len(lambda2_list)),lambda2_list,label="lambda2_lr")
plt.legend()
plt.savefig("LambdaLR.png")
plt.show()
