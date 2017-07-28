'''
#pandas基本介绍
import pandas as pd
import numpy as np
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
date = pd.date_range('20170728',periods = 6)
print(date)
df = pd.DataFrame(np.random.randn(6,4),index = date,columns = ['a','b','c','d'])
print(df)

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20170728'),
                    'C':pd.Series(1,index=list(range(4)),dtype = 'float32'),
                    'D':np.array([3]*4,dtype = 'int32'),
                    'E':pd.Categorical(["test","train","test","train"]),
                    
                    'F':'foo'})
print(df2)
print("the next output")
print(df2.dtypes)
print("the next output")
print(df2.index)
print("the next output")
print(df2.columns)
print("the next output")
print(df2.values)
print("the next output")
#df2.describe() 只能描述数字列的信息
print(df2.describe())
print("the next output")
print(df2.T)
print("the next output")
print(df2.sort_index(axis = 1,ascending = False))
print(df2.sort_index(axis = 0,ascending = False))
print(df2.sort_values(by = 'E'))
'''

#pandas选择数据
