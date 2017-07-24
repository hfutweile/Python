'''
#numpy属性
#ndim：维度
#shape：行数和列数
#size：元素个数
import numpy as np

array = np.array([[1,2,3],[2,3,4]])
print(array)
print("number of dim:",array.ndim)
print("size:",array.size)
print("shape:",array.shape)
'''

#numpy创建array
import numpy as np

#定义一维矩阵
a = np.array([2,23,4],dtype = np.int64)
#windows默认是int32,dtype int 可以指定int32或者int64
a = np.array([2,23,4],dtype = np.float)
#windows默认是float64,dtype float 可以指定是float32或者float64
print(a.dtype)

#定义二维矩阵
b = np.array([[2,23,4],[2,32,4]])
print(b)

c = np.zeros((3,4))
print(c)

d = np.ones((3,4),dtype = np.int16)
print(d)

e = np.empty((3,4))
print(e)

f = np.arange(10,20,2)
print(f)
f = np.arange(12).reshape((3,4))
print(f)
f = np.linspace(1,10,6).reshape((2,3))
print(f)
