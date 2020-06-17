# coding:utf-8

import numpy
import matplotlib.pyplot as plt

def loadDataSet():
    """

    :return:
    """
    dataSet = []
    fileName = "testSet.txt"
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet


def calDistance(instance1, instance2):
    """

    :param instance1:
    :param instance2:
    :return:
    """
    return numpy.sqrt(numpy.sum(numpy.power(instance1 - instance2, 2)))


def randomCenter(dataMat, k):
    """

    :param dataMat:
    :param k:
    :return:
    """
    numpy.random.seed(1111)
    n = numpy.shape(dataMat)[1]
    centroids = numpy.mat(numpy.zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(numpy.max(dataMat[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * numpy.random.rand(k, 1)
    return centroids


def kmeans(dataSet, k):
    """

    :param dataSet:
    :param k:
    :return:
    """
    dataMat = numpy.mat(dataSet)
    m = numpy.shape(dataMat)[0]
    clusterAssment = numpy.mat(numpy.ones((m, 2)))
    centroids = randomCenter(dataMat, k)
    print(centroids)
    printScatter(dataMat, centroids, clusterAssment)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = numpy.inf
            minIndex = -1
            for j in range(k):
                distJI = calDistance(centroids[j, :], dataMat[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 打印此时的图像
        printScatter(dataMat, centroids, clusterAssment)
        for cent in range(k):
            ptsInClust = dataMat[numpy.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(ptsInClust) != 0:
                centroids[cent, :] = numpy.mean(ptsInClust, axis=0)
        print(centroids)
    print(centroids)
    print(clusterAssment)
    printScatter(dataMat, centroids, clusterAssment)
    return centroids, clusterAssment


def printScatter(dataMat, centroids, clusterAssment):
    """

    :param dataMat:
    :param centroids:
    :param clusterAssment:
    :return:
    """
    dataSet = dataMat.A
    lenth = len(dataSet)
    labelMat = []
    for i in range(0, lenth):
        labelMat.append(clusterAssment[i, 0] + 1)

    fig = plt.figure()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1], 10.0 * numpy.ones(len(labelMat)), 15 * numpy.array(labelMat))
    ax.scatter(centroids.A[:, 0], centroids.A[:, 1], 50.0 * numpy.ones(len(centroids.A)), 15 * numpy.array([1,2,3,4]), edgecolors='red', marker='p', linewidths=2)
    plt.show()




dataSet = loadDataSet()
kmeans(dataSet, 4)


