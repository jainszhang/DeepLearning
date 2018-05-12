import pandas as pd
import numpy as np

s = pd.Series([1,2,3,4,5,np.nan,6])#创建list
#print(s)

dates = pd.date_range('20181101',periods=6)
# print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])#index为行，column为列

df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
# print(df1)
df2 = pd.DataFrame({'A' : 1.,
                       'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo'})
# print(df2.dtypes)#查看数据类型--每列数据类型
# print(df2.index)#打印行序列
# print(df2.columns)#打印列的名字

# print(df2.values)#打印它的值
# print(df2.describe())#打印表的描述，例如数量，均值，标准差，最小值等等

# print(df2.T)#转置

# print(df2.sort_index(axis=1,ascending=False))#排序--按照列--倒序

print(df2.sort_values(by='E'))#按照值进行排序








