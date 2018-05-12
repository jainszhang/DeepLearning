import pandas as pd
import numpy as np

#连接数据---标签都一致
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#合并方式1:axis=0代表行，axis=1代表列
res = pd.concat([df1,df2,df3],axis=0)

#忽略掉重复行号，自己重新编码
res1 = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
# print(res)



#合并方式2---列表头不一致
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])



#默认连接方式，没有的就用nan填充
res = pd.concat([df1, df2], axis=1, join='outer')
#连接相同的部分，没有的部分忽略掉
res = pd.concat([df1, df2], axis=0, join='inner',ignore_index=True)
# print(res)

#特殊处理--
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
# print(res)



#追加数据--append只能按照行添加axis=0

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])



res = df1.append(df2, ignore_index=True)
#追加两行
res = df1.append([df2, df3],ignore_index=True)


#一行一行添加---数据来一行添加一行
df1 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)

print(res)