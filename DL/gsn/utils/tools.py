import numpy as np
import os

def get_nb_files(input_dir):
    list_files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]
    return len(list_files)

def ComputePsnr(error=0.016432):
    psnr = 10*np.log10(255*255 / error) / 3.0
    print(psnr)

def FusionFeature(inputFeaturesDir,outputFeatureDir="./fusion_data/"):
    _dirNameList = os.listdir(inputFeaturesDir)
    _dirNameList.sort()#代表文件夹名字
    _fileNameList = []#表示某一个文件夹下所有文件名，不包含路径
    _fileLen = get_nb_files(inputFeaturesDir + _dirNameList[0] + '/')#某一种特征的文件夹中文件个数

    for i in range(len(_dirNameList)):
        assert _fileLen == get_nb_files(inputFeaturesDir + _dirNameList[i] + '/')#确保每个特征文件夹下的文件个数一致
        _fileNameList.append(os.listdir(inputFeaturesDir+_dirNameList[i]+'/'))
        _fileNameList.sort()



    _features = []#用来保存多个相同文件名的特征
    for i in range(len(_fileNameList[0])):
        _fileNameTmp = _fileNameList[0][i]#以第一个文件夹内的特征名作为标准对比后续读取的文件名是否一致
        for j in range(len(_dirNameList)):
            assert _fileNameTmp == _fileNameList[j][i]
            _tmpFileName = inputFeaturesDir+_dirNameList[j]+'/'+_fileNameList[j][i]
            _tmpData = np.load(_tmpFileName)
            _features.append(_tmpData)
        #求平均值
        print("step {}".format(i/1000))
        _featureArray =  np.array(_features).mean(axis=0)
        _features.clear()#_feature必须清空，不然持续添加
    #     保存特征平均值
        _outFileName = outputFeatureDir + _fileNameList[0][i]
        np.save(_outFileName,_featureArray)


    # print(_fileNameList[0])




inputFeaturesDir = "/home/jains/datasets/gsndatasets/fusion_feature/"
outputFeatureDir = "/home/jains/datasets/gsndatasets/fusion_data/"
FusionFeature(inputFeaturesDir=inputFeaturesDir,outputFeatureDir=outputFeatureDir)

