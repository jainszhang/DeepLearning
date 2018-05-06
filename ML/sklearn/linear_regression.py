from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#创造数据
# x,y = datasets.make_regression(n_samples=100,
#                                n_features=1,n_targets=1,noise=5)
# plt.scatter(x,y)
# plt.show()



loaded_data = datasets.load_boston()

data_x = loaded_data.data
data_y = loaded_data.target

print(data_x.shape)
model = LinearRegression()
model.fit(data_x,data_y)

print(model.coef_)#输出权重
print(model.intercept_)#输出偏置
print(model.score(data_x,data_y))#对预测结果与实际结果对比打分

