from sklearn import svm
from sklearn  import datasets
clf = svm.SVC()
iris = datasets.load_iris()
x,y = iris.data,iris.target

clf.fit(x,y)



#method 1:pickle

# import pickle
# #保存模型
# # with open('save/clf.pickle','wb') as f:
# #     pickle.dump(clf,f)
#
# #加载模型
# with open('save/clf.pickle','rb') as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(x[0:1]))






#methold 2:joblib
from sklearn.externals import joblib

#保存模型
#joblib.dump(clf,'save/clf.pkl_job')

#加载
clf3 = joblib.load('save/clf.pkl_job')
print(clf3.predict(x[0:1]))