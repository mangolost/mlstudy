# coding:utf-8

import logging
import random
import numpy
import matplotlib.pyplot as plt
from kNN.film import Film

logger = logging.getLogger(__name__)


class CategoryService:
    K = 20
    films = []

    def __init__(self, K=20, films=None):
        self.K = K
        if films is None:
            self.films = genDataSet()
        else:
            self.films = films

    def calType(self, film):
        """
        计算类型
        :param film:
        :return:
        """
        # 计算距离
        kiss_num = film.kiss_num
        fight_num = film.fight_num
        distances = []
        for item in self.films:
            distance = {"film": item, "value": (kiss_num - item.kiss_num) * (kiss_num - item.kiss_num) + (
                    fight_num - item.fight_num) * (
                                                       fight_num - item.fight_num)}
            distances.append(distance)
        # 排序
        distances.sort(key=lambda x: x["value"])
        # 取前K个
        distances = distances[0:self.K]
        # 计算前K个中各类型个数
        map = {}
        for distance in distances:
            type = distance["film"].type
            if type in map:
                map[type] = map[type] + 1
            else:
                map[type] = 1
        # 选出个数最多的类型
        max = 0
        type = ""
        for key in map:
            if map[key] > max:
                max = map[key]
                type = key
        return type


def genDataSet():
    """
    随机生成数据集
    :return:
    """

    seedNo = 1024
    random.seed(seedNo)

    filmSet = []
    for i in range(100):

        film = Film()
        film.id = i + 1
        film.name = "影片" + str(film.id)
        film.kiss_num = random.randint(0, 100)
        film.fight_num = random.randint(0, 100)
        if film.fight_num >= 2 * film.kiss_num:
            film.type = "武打剧"
        elif film.fight_num <= 0.5 * film.kiss_num:
            film.type = "言情剧"
        else:
            random_value = random.random()
            if random_value > 0.5:
                film.type = "武打剧"
            else:
                film.type = "言情剧"
        filmSet.append(film)
    printDataSet(filmSet)
    drawDataSet(filmSet)
    return filmSet


def printDataSet(filmSet=None):
    """
    打印生成的数据集
    :return:
    """
    if filmSet is None:
        filmSet = []
    for item in filmSet:
        print(item.__dict__)


def drawDataSet(filmSet=None):
    """
    绘制生成的数据集散点图
    :return:
    """
    if filmSet is None:
        return
    lenth = len(filmSet)
    dataMat = numpy.zeros((lenth, 2))
    labelMat = []
    for i in range(0, lenth):
        item = filmSet[i]
        dataMat[i, 0] = item.fight_num
        dataMat[i, 1] = item.kiss_num
        if item.type == '武打剧':
            labelMat.append(1)
        else:
            labelMat.append(2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], 15.0 * numpy.array(labelMat), 15.0 * numpy.array(labelMat))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
