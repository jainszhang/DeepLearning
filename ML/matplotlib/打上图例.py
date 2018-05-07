import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = 5*x

plt.figure()
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
            [r'$really\ bad$',r'$bad here \alpha$','normal','good','really good'
            ])


#在线上打上图例
# l1, = plt.plot(x,y2,label='up')
# l2, = plt.plot(x,y1,c='red',label='down')
# plt.legend(handles=[l1,l2,],labels=['aaa','bbb'],loc='best')#可更改线条图例的名字

plt.plot(x,y2,label='up')
plt.plot(x,y1,c='red',label='down')
plt.legend(loc='best')

plt.show()