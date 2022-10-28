#  基于BP神经网络的股票价格预测
&nbsp;&nbsp;&nbsp;&nbsp;通过BP神经网络对明尼亚波利斯春小麦期货的收盘价进行了预测，准确的预测出收盘价的变动。  
&nbsp;&nbsp;&nbsp;&nbsp;BP神经网络中采用优化器ADAM、学习率变化策略Warmup和余弦退化技术，可以有效的解决训练时间太长的的问题。  
&nbsp;&nbsp;&nbsp;&nbsp;采用Dropout机制可以有效的提升模型预测的准确率和鲁棒性。

#### 学习率采用warmup和CosineAnnealingLR效果图
![image](https://user-images.githubusercontent.com/82042336/198598120-c84a906a-c283-45fe-b24a-dc4828b20beb.png)

#  论文
#### 第1章  绪论  
#### 第2章  股票预测理论和方法  
#### 第3章  BP神经网络  
#### 第4章 神经网络的搭  
&nbsp;&nbsp;&nbsp;&nbsp;4.1 基于BP神经网络模型的建立  
&nbsp;&nbsp;&nbsp;&nbsp;4.2 网络结构设置  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.1 网络层数和神经元个数  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.2 激活函数  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.3 优化器  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.4 学习率变化策略  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.5 Dropout和WarmUp机制  
&nbsp;&nbsp;&nbsp;&nbsp;4.2.6 初始化参数的选取  
&nbsp;&nbsp;&nbsp;&nbsp;4.3 网络性能设置  
&nbsp;&nbsp;&nbsp;&nbsp;4.4 模型评价  
#### 第5章  基于BP神经网络的股票价格预测  
&nbsp;&nbsp;&nbsp;&nbsp;5.1 数据采集  
&nbsp;&nbsp;&nbsp;&nbsp;5.1.1 数据来源平台  
&nbsp;&nbsp;&nbsp;&nbsp;5.1.2 数据介绍  
&nbsp;&nbsp;&nbsp;&nbsp;5.2 数据预处理  
&nbsp;&nbsp;&nbsp;&nbsp;5.3 模型训练  
&nbsp;&nbsp;&nbsp;&nbsp;5.4 实验结果对比  
&nbsp;&nbsp;&nbsp;&nbsp;5.6 本章小结  
#### 第6章  总结与展望  
&nbsp;&nbsp;&nbsp;&nbsp;6.1 总结  
&nbsp;&nbsp;&nbsp;&nbsp;6.2 展望  
#### 参考文献  
#### 致    谢  
#### 附录A  外文原文  
#### 附录B  外文原文  
