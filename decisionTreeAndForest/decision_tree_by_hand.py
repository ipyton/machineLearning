import numpy  as np
from collections import defaultdict

class DecisionTree(object):
    def __init__(self):
        pass


    # 计算概率
    def _getDistribution(self):
        distribution = defaultdict(float)
        m, n = np.shape(data_array)
        for line in data_array:
            print(line[-1])
            distribution[line[-1]] += 1.0/m
        return distribution

    # calculate the entropy
    def _entropy(self):
        ent = 0.0
        distribution = self._getDistribution(data_array)

    # condition entropy
    def _condition_entropy(self):


    def _info_gain(self):


    def _choose_best_prop(self):


    def _split_data(self):

    def createTree(self):




def loadData():
    dataMat = []
    fr = open("decisiontree.txt")

    lines = fr.readlines()
    for line in lines:
        curLine = line.strip().split('\t')
        dataMat.append(curLine)

    return dataMat

if __name__ == "__main__":
    data = loadData()
    data_array = np.array(data)
    dt = DecisionTree()
    tree = dt.createTree(data_array)
    print(tree)



