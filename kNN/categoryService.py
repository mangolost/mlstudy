import logging
import random
from kNN.film import Film

logger = logging.getLogger(__name__)


def genDataSet():
    filmSet = []
    for i in range(200):
        filmx = Film()
        filmx.kiss_num = random.randint(0, 100)
        filmx.fight_num = random.randint(0, 100)
        if filmx.fight_num >= 2 * filmx.kiss_num:
            filmx.type = "动作片"
            filmSet.append(filmx)
        elif filmx.fight_num <= 0.5 * filmx.kiss_num:
            filmx.type = "爱情片"
            filmSet.append(filmx)
    return filmSet


class CategoryService:
    K = 20
    films = []

    def __init__(self, K=20, films=None):
        if films is None:
            films = []
        self.K = K
        self.films = films
        self.initiate()

    def initiate(self):
        self.films = genDataSet()

    def calType(self, film):
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
