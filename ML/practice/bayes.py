import numpy as np
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
dataSet1,label = loadDataSet()#获取源数据和对应标签
data = createVocabList(dataSet1)#提取出源数据中所有属性保存下来
#将所有训练数据向量化
trainData = []
for item in dataSet1:
    trainData.append(setOfWords2Vec(data,item))
#求y不同分类的先验概率
y = [label.count(0)/len(label),label.count(1)/len(label)]
y = np.array(y)
#分离不同分类下的训练数据
trainData = np.array(trainData)
train0 = []
train1 = []
for i in range(len(label)):
    if label[i]:
        train1.append(trainData[i])
    else:
        train0.append(trainData[i])
train0 = np.array(train0)#分类为0的训练数据
train1 = np.array(train1)#分类为1的训练数据
# #计算分类为0和1当中每个特征所占概率,P(Xi | yi)
count0 = []
count1 = []
for i in range(train0.shape[1]):#在y=0的情况下，遍历每一列，计算每个特征出现的次数
    count0.append(train0[:,i].tolist().count(1))
for i in range(train1.shape[1]):
    count1.append(train1[:,i].tolist().count(1))
p0 = np.array(count0) / train0.reshape((-1)).tolist().count(1)#计算在y=0情况下，每个特征所占y=0情况下所有特征的比例
p1 = np.array(count1) / train1.reshape((-1)).tolist().count(1)

#给定数据，做出预测
test = ['love','my','dalmation']
test1 = ['stupid','garbage']
v1 = setOfWords2Vec(data,test1)
score = [sum(v1 * p0) + np.log(y[0]),sum(v1 * p1) + np.log(y[1])]
print(np.argmax(score))