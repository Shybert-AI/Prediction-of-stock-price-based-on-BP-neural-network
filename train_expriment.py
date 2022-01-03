# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim.lr_scheduler as lrs
from lr_scheduler import GradualWarmupScheduler
import torch.optim as optim   
from option import get_args   #训练参数

'''-------------------------1、数据处理-------------------------'''
#Date：时间
#open：开盘价
#High：最高价
#Low：最低价
#Last：收盘价
#Volume：成交量
#Low：最低价
#Open Interest:未平仓合约
# 1、数据分析
# dataset: 6462行数据
# 收盘价分析：
# <300  300-400   400-500   500-600   700-800  >800
# 7     2164        1589      710       472    780
# 用来训练 5600/16=350组
# 用来测试 848/16 = 53组
#
# 以15天为一组，预测下1天的收盘价

stockFile = 'CHRIS-MGEX_MW3.csv'
df = pd.read_csv(stockFile, index_col=0, parse_dates=[0])   # index_col:指定某列为索引  parse_dates:将某列解析为时间索引
data_last = df['Last'].values   # 将数据转为numpy.ndarray类型
data_last = data_last.tolist()  # 将数据转为list类型

# 切分训练集，测试集
x_train, y_train, x_test, y_test = [], [], [], []
temp_list = []

# 将前5600天为训练集数据，剩下的是测试集
for id, i in enumerate(data_last):
    if id+1 <= 5600:    # 前5600行是训练数据
        if (id+1) % 16 != 0:
            temp_list.append(i)
        else:
            x_train.append(temp_list)
            temp_list = []
            y_train.append([i])
    elif id+1 <= 6448:
        if (id + 1) % 16 != 0:
            temp_list.append(i)
        else:
            x_test.append(temp_list)
            temp_list = []
            y_test.append([i])

# 将数据转为 tensor类型
# 训练数据：15天一组，350组
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
# 测试数据：15天一组，53组
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

'''-------------------------2、训练模型-------------------------'''
# 超参数
args = get_args()
torch.manual_seed(args.seed)
#num_epochs = 100   # 训练批次
#batch_size = 10     # 10个样本一个batch—size
#learning_rate = 0.0003      # 学习率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 用gpu跑

# 制作dataset，dataloader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# 损失函数
criterion = nn.MSELoss()
# 模型
model = NeuralNet(15, 100, 6, 1,p=0,active_func="Relu").to(device)
# 梯度下降优化器和学习率变化策略
def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())  #筛选需要更新的参数（为True的参数）

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),#一阶动量，二阶动量  过去的占比 
            'eps': 1e-08
        }
        
    else:
        raise NameError('Not Supportes Optimizer')

    kwargs['lr'] = args.learning_rate
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer):
    if args.decay_type == 'step':    #阶跃变化
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif args.decay_type == 'step_warmup':  #预热阶跃变化
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler
        )
    elif args.decay_type == 'cosine_warmup':  #学习率先从很小的数值线性增加到预设学习率，然后按照cos函数值进行衰减
        cosine_scheduler = lrs.CosineAnnealingLR(   #
            optimizer,         #优化器
            T_max=args.epochs  #总共的迭代次数
        )
        scheduler = GradualWarmupScheduler(  #预热学习率 + cos变化
            optimizer,
            multiplier=1,
            total_epoch=args.epochs//10,  #预热的阶段  
            after_scheduler=cosine_scheduler #预热完怎么处理
        )
    else:
        raise Exception('unknown lr scheduler: {}'.format(args.decay_type))
    
    return scheduler
     
def init_weights(m):  #m为模块
    """初始化权重."""
    print(m)
    classname = m.__class__.__name__
    method = "normal_"  #"kaiming_normal_"  
    if classname.find('Linear') != -1:
        if method =="kaiming_normal_":     #恺明正态分布初始化
            m.weight.data.zero_()  #先将定义线性层时的权重置为零
            nn.init.kaiming_normal_(m.weight, mode='fan_in')    #loss 4.3867
            nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5)) #均匀分布 #loss=4.3823
            m.bias.data.zero_()    
        else:   
            m.weight.data.zero_()  #先将定义线性层时的权重置为零
            nn.init.normal_(m.weight, mean=0, std=0.1) #正态分布初始化   #4.4206
            #nn.init.uniform_(m.weight,b=0.1)               #均匀分布     #4.3127
            m.bias.data.zero_()
        
#model.apply(init_weights)
lr_change = args.lr_change
""" 定义优化器 """
optimizer = make_optimizer(args, model) if lr_change =="YES" else torch.optim.Adam(model.parameters(), lr=args.learning_rate)  #更新参数,学习率在优化器中

""" 定义学习率变化策略 """
if lr_change == "YES":
    scheduler = make_scheduler(args, optimizer) #改变学习率的模块  读学习率，赋予新值
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
epoch_train_loss = 0.0  # epoch 训练总损失
epoch_test_loss = 0.0   # epoch 测试总损失
total_train_length = len(train_loader)  # 训练集数量
total_test_length = len(test_loader)    # 测试集数量
test_label = []     # 保存测试集的标签
last_epoch_predict = []     # 最后一个批次预测结果
train_loss_list = []
test_loss_list = []
error_list = []
lr_list = [] #学习率
# 开始训练
for epoch in range(args.epochs):
    model.train()   #训练模式
    error = 0.0     # 测试预测误差
    train_loss = 0.0    # 当前epoch的 训练损失
    test_loss = 0.0     # 当前epoch的 测试损失
    for idx, (x, y) in enumerate(train_loader):
        # gpu加速
        x = x.to(device)
        y = y.to(device)

        # 前向传播
        output = model(x)
        train_loss = criterion(output, y)   # 当前损失
        epoch_train_loss += train_loss.item()   # epoch总损失

        # Backward and optimize
        optimizer.zero_grad()   # 梯度清0
        train_loss.backward()   # 反向传播
        optimizer.step()        # 更新梯度

    with torch.no_grad():
        model.eval()   #测试模式
        for idy, (x, y) in enumerate(test_loader):
            # gpu加速
            x = x.to(device)
            y = y.to(device)
            # 预测
            output = model(x)

            # predict_output += output.item()
            test_loss = criterion(output, y)    # 测试损失
            epoch_test_loss += test_loss.item()     # 计算 测试损失和
            error += math.fabs(y.item() - output.item())    # 误差计算公式：相减求绝对值

            # 保存最后一个批次预测结果，输出画图
            if epoch == args.epochs - 1:
                test_label.append(y.item())
                last_epoch_predict.append(output.item())

    epoch_train_loss /= total_train_length  # 计算epoch 平均训练损失
    epoch_test_loss /= total_test_length    # 计算epoch 平均测试损失
    error /= total_test_length              # MAE 评价均值误差
    train_loss_list.append(epoch_train_loss)  #统计训练损失
    test_loss_list.append(epoch_train_loss)   #统计测试损失
    error_list.append(error)                  #统计平均误差
    if lr_change == "YES":
        scheduler.step()                          #每次epoach都要更新学习率
        lr_list.append(scheduler.get_lr()[0])     #每个epoch的学习率
    print('Epoch [{}/{}], TrainLoss:{:.4f}, TestLoss: {:.4f}, TestError:{:.4f}'.format(epoch+1, args.epochs, epoch_train_loss, epoch_test_loss, error))

# 预测结果 画图
x = np.linspace(0, 53, 53)
#plt.plot(x, test_label, color='blue', marker='o')
#plt.plot(x, last_epoch_predict, color='red', marker='o')
#plt.xlabel('test data')
#plt.ylabel('last price')
#plt.legend(labels=['real', 'predict'])  # 加图例
#plt.savefig("预测结果.png")
#plt.show()


#损失图像
plt.subplot(3,1,1)
plt.plot(list(range(args.epochs)),train_loss_list,c="r",label="train_loss")
plt.plot(list(range(args.epochs)),test_loss_list,c="g",label="test_loss")
plt.legend()
#MAE 平均误差图像
plt.subplot(3,1,2)
plt.plot(error_list,c="r",label="MAE")
plt.text(len(error_list),error_list[-2]+3,f"MAE={error_list[-2]}" )
plt.legend()

plt.subplot(3,1,3)
plt.plot(x, test_label, color='blue', marker='o')
plt.plot(x, last_epoch_predict, color='red', marker='o')
plt.xlabel('test data')
plt.ylabel('last price')
plt.legend(labels=['real', 'predict'])  # 加图例

#plt.savefig("恺明正态分布初始化参数效果图.png")
#plt.savefig("正态分布初始化参数效果图.png")
#plt.savefig("恺明均价分布初始化参数效果图.png")
#plt.savefig("均价分布初始化参数效果图.png")

plt.show()

#学习率变化
#plt.plot(lr_list, c="b")
#plt.xlabel("epoch")
#plt.ylabel("LR")
#plt.savefig("LR.png")
#plt.show()