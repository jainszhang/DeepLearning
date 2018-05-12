import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#plot datad
data = pd.Series(np.random.randn(1000),index=np.arange(1000))

data = data.cumsum()




#DataFrame
data = pd.DataFrame(np.random.randn(1000,4),
                    index=np.arange(1000),
                    columns=list("ABCD"))


data = data.cumsum()
# data.plot()
# plt.show()
#plt.plot(x=,y=)
#pandas中data本身就是数据
# data.plot()
#
# plt.show()

# plot 方法
# bar,box,kde,area,scatter,hexbin,hist,pie
ax = data.plot.scatter(x='A',y='B',label='class1',c='r')

data.plot.scatter(x='A',y='C',label='class2',ax=ax,c='b')
plt.show()