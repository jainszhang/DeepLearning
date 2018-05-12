import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(data=np.random.randn(6,4), index=dates, columns=['A', 'B', 'C', 'D'])

#根据下标改变某一个的值
df.iloc[2,2] = 111

#根据标签改变某一个值
df.loc['201430101','B'] = 222

#根据标签+数字改变值
#df.ix[[:3],'A'] = 333

#根据条件改变值
df.A[df.A>0] = 0
df[df.A>0] = 0

#加入新的列
df['F'] = np.nan

#添加新的一列--
df['E'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods = 6))
print(df)