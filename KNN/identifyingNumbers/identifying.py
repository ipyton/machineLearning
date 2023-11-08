import os

import numpy as np

from KNN.basic.KNNByHand import *

def img2vector(fileName):
    returnVect = np.zeros((1,1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        if lineStr != "\n":
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def IdentImgClassTest():
    classLabels = []
    trainingFileList = os.listdir("")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        classLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./dat/TrainData/%s' % fileNameStr)
    testFileList = os.listdir('./dat/TestData')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector('./data/TestData/%s' % fileNameStr)

        classifierResult = classify(vectorUnderTest, trainingMat, classLabels, 3)
        print('number recognized is: %d, the real number is %d' % (classifierResult, classNumStr))

        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nerror tries %d" % errorCount)
    errorRate = errorCount / float(mTest)
    print("\n correct rate %f" % (1 - errorRate))

if __name__ == "__main__":
    IdentImgClassTest()


