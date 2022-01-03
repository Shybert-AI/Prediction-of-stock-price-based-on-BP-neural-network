# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

class Function(object):
    """
    功能：定义机器学习中常见的激活函数
    """
    def __init__(self,x):
        self.x = x
        
    def step_function(self):
        """阶跃函数以0为界，输出从0切换为1（或者从1切换为0）。
        它的值呈阶梯式变化，所以称为阶跃函数"""        
        return np.array(self.x > 0, dtype=np.int)
    
    
    def sigmoid(self):
        """
        sigmiod的的表达式为：y = 1/(1+e^-x)
        """
        return 1/(1 + np.exp(-self.x))
    
    def tanh(self):
        """
        tanh的表达式为： y = (e^x-e^-x)/(e^x+e^-x)
        """
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def relu(self):
        """
        relu的表达式为： y = x if x > 0 else 0
        """
        return np.maximum(x,0)
    
    def softplas(self):
        """
        softplas的表达式为： y = ln(1+e^x)
        """
        return np.log(np.exp(x)+1)
    
    def swish(self):
        """
        swish的表达式为： y = x*sigmiod(beta*x) 取beta参数为1
        """
        return x*(self.sigmoid())
    
    def mish(self):
        """
        mish的表达式为：  y = x*tanh(sortplas())
        """
        x1 = self.softplas()
        y = (np.exp(x1)-np.exp(-x1))/(np.exp(x1)+np.exp(-x1))
        
        return x*y
    def gelu(self):
        """
        GELU = x*1/2[1+erf(x/根号2)]
        """
        import math
        x1 = x/math.sqrt(2.0)
        y = (np.exp(x1)-np.exp(-x1))/(np.exp(x1)+np.exp(-x1))
        return x * 0.5 * (1.0 + y)
if __name__ == "__main__":
    x = np.linspace(-6, 6,100)
    F = Function(x)
    #1.sigmoid
    plt.plot(x,F.step_function(),label="step_function")
    plt.plot(x,F.sigmoid(),label="Sigmoid")
    plt.plot(x,F.tanh(),label="Tanh")
    plt.legend()
    plt.savefig("activefunction1.png")
    plt.show()
    ##2. tanh 
    #plt.plot(x,F.tanh(),label="Tanh")
    #plt.legend() 
    #plt.show()
    ##3. relu 
    #plt.plot(x,F.relu(),label="Relu")
    #plt.legend() 
    #plt.show()  
    #4. softplas 
    plt.plot(x,F.relu(),label="Relu")
    plt.plot(x,F.softplas(),label="SoftPlas")
    plt.plot(x,F.swish(),label="Swish")
    plt.plot(x,F.mish(),label="Mish")
    plt.plot(x,F.gelu(),label="GELU")
    plt.legend() 
    plt.savefig("activefunction2.png")
    plt.show() 
    ##5. swish 
    #plt.plot(x,F.swish(),label="Swish")
    #plt.legend() 
    #plt.show()    
    ##6. mish 
    #plt.plot(x,F.mish(),label="Mish")
    #plt.legend() 
    #plt.show()  
    ##6. gelu 
    #plt.plot(x,F.gelu(),label="GELU")
    #plt.legend() 
    #plt.show()     