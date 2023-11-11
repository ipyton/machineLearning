import operator

import numpy as np
import matplotlib as plt


def classify(normData, dataSet, labels, k): # top k nearest
    # calculate the eucler distance
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet
    sqlDiffMat = diffMat ** 2
    sqlDistances = sqlDiffMat.sum(axis=1)
    distance = sqlDistances ** 0.5

    sortedDistIndices = distance.argsort()
    classCount = {}

    # calculate k nearest to normData points in dataset and count them
    # this is a vote procedure
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# convert
def file_to_matrix(fileName):
    fr = open(fileName)

    linesArray = fr.readlines()
    numOfLines = len(linesArray)
    returnMat = np.zeros((numOfLines, 3))
    classLabelVector = []
    index = 0
    for line in linesArray:
        line = line.strip()
        print(line.split('\t'))
        listFromLine = list(map(float, line.split('\t')))
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# normalization the data in dataset: to avoid a column which has a bigger value which interfere with the training
# process maximum-minimum normalization
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # return min value of every column
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]

    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# import the dating dataset and classify the categories that this belongs to
def datingClassTest():
    hoRatio = 0.1

    # import the dataset
    datingDataMat, datingLabels = file_to_matrix("basic/data/datingTestSet2.txt")

    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4) # te
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    errorRate = errorCount / float(numTestVecs)
    return 1 - errorRate


# visualize
def createScatterDiagram():
    datingDataMat, datingLabels = file_to_matrix('basic/data/datingTestSet2.txt')
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    plt.figure()
    axes = plt.subplot(111)
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for i in range(len(datingLabels)):
        if datingLabels[i] == 1:
            type1_x.append(datingDataMat[i][0])
            type1_y.append(datingDataMat[i][1])
        if datingLabels[i] == 2:
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])
        if datingLabels[i] == 3:
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')

    plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=datingLabels)
    plt.show()

    plt.xlabel('aa')
    plt.ylabel('bb')

    plt.scatter()
    plt.show()


def classifyPerson():
    resultList = ["no sense", 'looks good', 'charm']
    input_man = [50000, 8, 9.5]
    datingDataMat, datingLabels = file_to_matrix("basic/data/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    result = classify((input_man - minVals) / ranges, normMat, datingLabels, 10)
    print("you will date", resultList[result - 1])


acc = datingClassTest()
if acc > 0.9:
    classifyPerson()
