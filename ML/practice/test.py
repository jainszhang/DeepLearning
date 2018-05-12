import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

train = pd.read_csv("./data_set/Titanic/train.csv")


print(train.head(2))
#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
#pclass:1-一等仓，2-二等舱，3-三等舱
#Cabin:代表船舱



#检查是否有缺失值--Age,Cabin,Embarked
# print(np.any(titanic['Cabin'].isnull()==True))

# train.Cabin[train.Cabin.isnull()==True] = 'U0'



#特征分析
#男女生存比例
# print(titanic.info())
# print(titanic.describe())#整体存活率0.3838
#
# female_num,male_num = x[x.Sex=='female'].iloc[:,0].size,x[x.Sex=='male'].iloc[:,0].size
# female_sur_num,male_sur_num = titanic[(titanic.Sex=='female') & (titanic.Survived==1)].iloc[:,0].size,\
#                               titanic[(titanic.Sex=='male') & (titanic.Survived==1)].iloc[:,0].size
# print(female_num,male_num)
# print(female_sur_num,male_sur_num)
# print(female_sur_num/female_num,male_sur_num/male_num)
#经过分析，女性存活率为0.7420,男性为0.1889

#分析舱位类型
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
#    Pclass  Survived
# 0       1  0.629630
# 1       2  0.472826
# 2       3  0.242363









label = train['Survived']
x = train.loc[:,['Pclass','Sex','Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]










