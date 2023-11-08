import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 12))


def loadDataSet(file_name):
    dataMat = []
    fr = open(file_name)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEuclid(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# get k points which is used to calculate at the first step.
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroid = np.mat(np.zeros((k, n)))

    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.max(dataSet[:, j]) - minJ

        centroid[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroid


def kMeans(dataSet, k, distMeans=distEuclid, createCent=randCent):  # find k
    m = np.shape(dataSet)[0]

    clusterAssment = np.mat(np.zeros((m, 2)))  # notice the difference between the mat and matrix (index, dist)
    centroid = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # all points
            minDist = np.inf
            minIndex = -1
            for j in range(k):  # selected points
                x = centroid[j, :]
                distJI = distMeans(x, dataSet[i, :])  # calculate all points to select point
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j  #
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        for cent in range(k):  # prepare for the next time iteration, calculate the mean position in same category
            # none zero returns a matrix of indices of given precondition
            # actually return the index of a specific category
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]

            centroid[cent, :] = np.mean(ptsInCluster, axis=0)
    return centroid, clusterAssment


if __name__ == '__main__':
    dataMat = np.mat(loadDataSet('data/testSet.txt'))
    k = 4
    centroid, clusterAssment = kMeans(dataMat, k, distMeans=distEuclid, createCent=randCent)
    print(clusterAssment)
    print(centroid)



