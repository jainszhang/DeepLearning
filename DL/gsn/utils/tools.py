import numpy as np
import os
def ComputePsnr(error=0.016432):
    psnr = 10*np.log10(255*255 / error) / 3.0
    print(psnr)

def FusionFeature(inputFeaturesDir,outputFeatureDir="./fusion_data/"):
    _dirNameList = os.listdir(inputFeaturesDir)
    _dirNameList.sort()
    _fileNameList = []
    for i in range(len(_dirNameList)):
        _fileNameList.append(os.listdir(inputFeaturesDir+_dirNameList[i]+'/'))
        _fileNameList.sort()

    _features = []
    for i in range(len(_fileNameList[0])):
        for j in range(len(_dirNameList)):
            _tmpFileName = inputFeaturesDir+_dirNameList[j]+'/'+_fileNameList[j][i]
            _tmpData = np.load(_tmpFileName)
            _features.append(_tmpData)
        #求平均值
        print("step {}".format(i/100))
        _featureArray =  np.array(_features).mean(axis=0)
    #     保存特征平均值
        _outFileName = outputFeatureDir + _fileNameList[0][i]
        np.save(_outFileName,_featureArray)

    # print(_fileNameList[0])




inputFeaturesDir = "/home/jains/datasets/gsndatasets/fusion_feature/"
outputFeatureDir = "/home/jains/datasets/gsndatasets/fusion_data/"
FusionFeature(inputFeaturesDir=inputFeaturesDir,outputFeatureDir=outputFeatureDir)

