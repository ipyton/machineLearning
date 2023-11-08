import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np


plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170  # something like seed

'''
n_samples:numbers of records
n_features: dimension of data
centers: the center of data generated
shuffle: shuffle the cards
random_state:seed
'''
# make some blobs and return the results
x, y = make_blobs(n_samples=n_samples, random_state=random_state)
print(x, y)


y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)

plt.subplot(221)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("kmeans01")
plt.savefig("kmeans01.png")



# transform and calculate
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
x_aniso = np.dot(x, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x_aniso)
plt.subplot(222)
plt.scatter(x_aniso[:, 0], x_aniso[:, 1], c=y_pred)
plt.title("kmeans02")


# x[y==0] means get elements from x and it is related to y
x_filtered = np.vstack((x[y == 0][:500], x[y == 1][:100], x[y == 2][:10]))


y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x_filtered)
plt.subplot(223)
plt.scatter(x_filtered[:,0], x_filtered[:, 1], c=y_pred)
plt.title("kmeans03")


# uses data in file to build the model
dataMat = []
fr = open("data/testSet.txt", "r")

for line in fr.readlines():
    if line.strip() != "":
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)

dataMat = np.array(dataMat)
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(dataMat)
print(y_pred)

plt.subplot(224)
plt.scatter(dataMat[:, 0], dataMat[:, 1], c=y_pred)
plt.title("kmeans4")
plt.show()


