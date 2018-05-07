import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = 5*x

#定了figure以后都是跟这个figure有关的
# plt.figure()
# plt.plot(x,y1)

plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,c='red')


#设置坐标轴
plt.xlim((-1,2))#轴范围
plt.ylim((-2,3))
plt.xlabel("I am x")#标签名
plt.ylabel("I am y")



#给x轴设置角标（单位）
new_ticks = np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)



#用其他字符替换掉数字单位 +++ 改变字体
plt.yticks([-2,-1.8,-1,1.22,3],
            [r'$really\ bad$',r'$bad here \alpha$','normal','good','really good'
            ])



#修改坐标原点位置
#gca = get current axis
ax = plt.gca()
ax.spines['right'].set_color('none')#设置右边的轴消失
ax.spines['top'].set_color('none')#设置上面的轴消失


#上面还没有设置默认的x，y轴
#此处设置那个为x轴哪个为y轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#挪动x，y轴坐标中心位置
ax.spines['bottom'].set_position(('data',0))#横坐标是纵向轴的值-1------axes表示定位在多少百分位置
ax.spines['left'].set_position(('data',0))#y轴坐标是横坐标的值0




plt.show()