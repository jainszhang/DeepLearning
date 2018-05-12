import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan


#按照行 / 列 丢掉
# print(df.dropna(axis=1,how='any'))#how = any / all--一行都是nan是会丢掉


##填充丢失数据
# print(df.fillna(value=0))

#查看是否为缺失数据
print(df.isnull())

#检查是否会有一个为nan，是的话返回True
print(np.any(df.isnull())==True)