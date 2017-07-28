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
'''

'''
#numpy基础运算1

import numpy as np


a = np.array([10,20,30,40])
b = np.arange(4)
print(a)
print(b)

#按位运算
print(b<3)
print(b==3)

c = a - b
print(c)

c = b ** 2
print(c)

c = 10 * np.sin(a)
print(c)

d = np.array([[1,1],[0,1]])
e = np.arange(4).reshape((2,2))
print(d,"\n",e)

f = d * e           #按位相乘
g = np.dot(d,e)     #矩阵相乘
g_dot = d.dot(e)
print(f)
print(g)
print(g_dot)

a = np.random.random((2,4))
print(a)
print(np.sum(a))
#axis = 1 在行中进行运算，axis = 0在列中进行运算
print("axis")
print(np.max(a,axis = 1))
print(np.max(a,axis = 0))
print("axis")
print(np.max(a))
print(np.min(a))
'''

'''
#numpy基础教学2
import numpy as np

A = np.arange(14,2,-1).reshape((3,4))
print(A)
#np.argmin()求元素值最小的索引
#np.argmax()求元素值最大的索引
#np.mean()求矩阵元素值的平均值
#np.average()求矩阵元素的平均值（老版本指令）
#np.median()求矩阵元素值的中位数
#np.cumsum()求元素前n项值不断累加到第n项的结果
#np.diff()每行相邻两项的差
#np.nonzero()矩阵中非零值的行列索引
#np.sort()每行排序
#np.transpose() 求矩阵的转置
#A.T求A的转置
#np.clip(A,MIN,MAX)将矩阵的元素值限制在MIN和MAX之间，大于MAX为MAX，小于MIN为MIN
#OR   A.mean() ...
print(np.argmin(A))
print(np.argmax(A))
print(np.mean(A))
print(np.median(A))
print(np.cumsum(A))
print(np.diff(A))
print(np.nonzero(A))
print(np.sort(A))
print(A.T)
print(np.transpose(A))
print(A.T.dot(A))
print(np.clip(A,5,9))
'''

'''
#numpy索引
import numpy as np

A = np.arange(3,15).reshape((3,4))
print(A)
#A[n] A[n,:] 第n行
#A[:,n]第n列
#A[m,n] A[m][n]第m行n列的数
#A[n,m:l] 矩阵第n行第m到l-1的元素值
print(A[2])
print(A[1][2])
print(A[1,2])
print(A[2,:])
print(A[2,1:3])

#迭代矩阵的行
for row in A:
    print(row)

#迭代矩阵的列
for column in A.T:
    print(column)

#迭代矩阵所有元素
#A.flatten()将多维矩阵转变为一维矩阵并返回该矩阵
#A.flat将多维矩阵转换为一维矩阵
print(A.flatten())
for item in A.flat:
    print(item)
'''

'''
#numpy array合并
import numpy as np
#np.vstack() vertical stack 上下合并
#np.hstack() horizontal stack左右合并

A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A,B))    
D = np.hstack((A,B))    
print("C:",C)
print("D:",D)
print(C.shape,D.shape)

#无法使用转置将一个序列变成一个矩阵,无法改变行列矩阵
print("A.T:",A.T)
#np.newaxis指在对应行或者列上排列
print(A[:,np.newaxis])
C = A[:,np.newaxis]
D = B[:,np.newaxis]
E = np.hstack((C,D))
print(E)

#np.concatenate() 多个矩阵纵向或者横向的合并
F = np.concatenate((C,D,C,D),axis = 1)
print(F)
'''
'''
#numpy array分割
import numpy as np

A = np.arange(12).reshape((3,4))
print(A)

#np.split() 对矩阵进行分割，axis = 1 对列操作，axis = 0 对行操作,必须进行等量分割
print(np.split(A,2,axis = 1))
print(np.split(A,3,axis = 0))
#np.array_split() 可以进行不等量分割，但是似乎不可控哪些分割在一起
print(np.array_split(A,3, axis = 1))
#np.vsplit() 纵向分割
#np.hsplit() 横向分割
print(np.vsplit(A,3))
print(np.hsplit(A,2))
'''

#numpy copy & deep copy
import numpy as np
a = np.arange(4)
b = a
c = a
d = b
a[0] = 4
print(a)
print(b,c,d)
#在Python中赋值操作的变量是同一个变量，类似于c语言中的引用
print(b is a,c is a,d is a)
d[1:3] = [22,33]
print(d,a)

#a.copy() deep copy,将a的值赋给b，但是a，b没有关联
b = a.copy()
print(a is b)
