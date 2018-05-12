import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
# print(df['A'], df.B)#两种方式打印莫一列数据
# print(df[0:3], df['20130102':'20130104'])#两种方式打印莫一行


#通过标签打印数据
# print(df.loc['20130101'])
# print(df.loc[:,['A','B']])#打印所有行中，某一些列数据    print(df.loc['20130102',['A','B']])


#纯数字筛选
# print(df.iloc[3:5,1:3])
# print(df.iloc[[1,3,4],1:3])


#标签+数字筛选
# print(df.ix[:3,['A','C']])


#进行是否条件的筛选
print(df[df.A>1])