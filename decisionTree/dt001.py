from math import log
from matplotlib import pyplot
import operator
import pickle

# 支持中文处理
pyplot.rcParams['font.sans-serif'] = ['SimHei']
pyplot.rcParams['axes.unicode_minus'] = False

fr = open('lense.txt')
dataSet = [record.strip().split('\t') for record in fr.readlines()]
labels = ['age', 'prescript', 'astigmatic', 'tear_rate']


def calShannonEnt(dataSet):
    """
    计算数据集的香农熵
    :param dataSet: 数据集
    :return: 香农熵
    """
    lenth = len(dataSet)
    categoryCounts = {}  # 分类值出现次数
    for data in dataSet:
        currentLabel = data[-1]  # 获取数据集最后一列，即分类值
        if currentLabel not in categoryCounts.keys():
            categoryCounts[currentLabel] = 1
        else:
            categoryCounts[currentLabel] += 1
    shannonEnt = 0
    for key in categoryCounts:
        prob = float(categoryCounts[key]) / lenth
        shannonEnt += -prob * log(prob, 2)  # 香农熵计算公式
    return shannonEnt


shannonEnt = calShannonEnt(dataSet)
print(shannonEnt)


def splitDataSet(dataSet, index, value):
    """
    将数据集按照某个特征的某个值获取子数据集
    :param dataSet: 原数据集
    :param index: 数据集某个特征在数据中的index
    :param value: 该特征中用于划分的值
    :return: 子数据集
    """
    subDataSet = []
    for data in dataSet:
        if data[index] == value:  # 选择复合筛选的数据
            reducedData = data[:index]  # 选择0到index-1列
            reducedData.extend(data[index + 1:])  # 加上index+1列以后，这样去掉了第index列
            subDataSet.append(reducedData)  # 加入子数据集
    return subDataSet


print(splitDataSet(dataSet, 0, "young"))
print(splitDataSet(dataSet, 0, "pre"))
print(splitDataSet(dataSet, 0, "presbyopic"))
print(splitDataSet(dataSet, 1, "myope"))
print(splitDataSet(dataSet, 1, "hyper"))
print(splitDataSet(dataSet, 2, "no"))
print(splitDataSet(dataSet, 2, "yes"))
print(splitDataSet(dataSet, 3, "reduced"))
print(splitDataSet(dataSet, 3, "normal"))


def chooseBestFeatureToSplit(dataSet):
    """
    针对一个数据集，选择最好的划分特征
    :param dataSet: 数据集
    :return: 最好的划分特征的index
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数
    baseShannonEnt = calShannonEnt(dataSet)  # 数据集初始香农熵
    bestInfoGain = 0  # 最佳信息增益，初始化为0
    bestFeature = -1  # 最佳划分特征index，初始化为-1
    for i in range(numFeatures):  # 遍历每个特征
        featureValueList = [data[i] for data in dataSet]  # 获取该特征下所有可能的特征值(有重复)
        uniqueValues = set(featureValueList)  # 特征值去重
        newShannonEnt = 0  # 该特征下各种分划的总香农熵，初始化为0
        for value in uniqueValues:  # 遍历每种特征值
            subDataSet = splitDataSet(dataSet, i, value)  # 获取在该特征与该特征值下的子数据集
            prob = float(len(subDataSet)) / len(dataSet)  # 获取该子数据集在该特征所有分化的数据集中的概率
            newShannonEnt += prob * calShannonEnt(subDataSet)  # 该特征下各种分划的总香农熵，为每个数据集的香农熵与该数据集概率乘积的累积和
        infoGain = baseShannonEnt - newShannonEnt  # 该特征下的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i  # 选择信息增益最大的特征
    return bestFeature


print(chooseBestFeatureToSplit(dataSet))


def majorityCnt(classList):
    """
    分类列表中选出次数最多的列表元素值
    :param classList: 分类列表
    :return: 选出的分类
    """
    classCnt = {}
    for item in classList:
        if item not in classCnt.keys():
            classCnt[item] = 0
        classCnt[item] += 1
    sortedClassCnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)  # 次数从多到少排序
    return sortedClassCnt[0][0]


def createTree(dataSet, labels):
    """
    根据数据集创建决策树
    :param dataSet: 数据集
    :param labels: 特征文字说明列表
    :return: 决策树
    """
    classList = [data[-1] for data in dataSet]  # 分类列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果只有一个分类，则返回分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 如果特征已经用完仍有多个分类，则计算次数最多的那个分类
    bestFeature = chooseBestFeatureToSplit(dataSet)  # 按照信息增益最大化选择最佳特征
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}  # 初始化决策树
    featureValueList = [data[bestFeature] for data in dataSet]
    uniqueValues = set(featureValueList)  # 特征值去重
    for value in uniqueValues:  # 对每个特征值递归构建子决策树
        subLables = labels[:]
        del subLables[bestFeature]
        subDataSet = splitDataSet(dataSet, bestFeature, value)
        subTree = createTree(subDataSet, subLables)  # 对构建子决策树
        myTree[bestFeatureLabel][value] = subTree
    return myTree


decisionTree = createTree(dataSet, labels)
print(decisionTree)


def getLeafNum(myTree):
    """
    获取决策树叶节点个数
    :param myTree: 决策树
    :return: 叶节点个数
    """
    num = 0
    firstKey = list(myTree.keys())[0]
    subDict = myTree[firstKey]
    for key in subDict.keys():
        if isinstance(subDict[key], dict):   # 判断是否为字典类型
            subNum = getLeafNum(subDict[key])   # 递归计算子树叶结点个数
            num += subNum
        else:
            num += 1
    return num


def getTreeDepth(myTree):
    """
    获取决策树深度
    :param myTree: 决策树
    :return: 深度
    """
    maxDepth = 0
    firstKey = list(myTree.keys())[0]
    subDict = myTree[firstKey]
    for key in subDict.keys():
        if isinstance(subDict[key], dict):  # 判断是否为字典类型
            subDepth = 1 + getTreeDepth(subDict[key])  # 递归计算子树叶深度
        else:
            subDepth = 1
        if subDepth > maxDepth:
            maxDepth = subDepth
    return maxDepth


print(getLeafNum(decisionTree))
print(getTreeDepth(decisionTree))

# 设置决策节点、叶结点、箭头样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="circle", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def createPlot(inTree):
    """
    创建决策树图形
    :param inTree: 决策树
    :return:
    """
    fig = pyplot.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = pyplot.subplot(111, frameon="false", **axprops)
    plotTree.totalW = float(getLeafNum(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")
    pyplot.show()


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树（子树)
    :param myTree: 决策树
    :param parentPt: 父节点坐标
    :param nodeTxt: 结点文本
    :return:
    """
    leafNum = getLeafNum(myTree)
    # depth = getTreeDepth(myTree)
    firstKey = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1 + leafNum) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstKey, cntrPt, parentPt, decisionNode)
    subDict = myTree[firstKey]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in subDict.keys():
        if isinstance(subDict[key], dict):
            plotTree(subDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(subDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点（以及箭头)
    :param nodeTxt: 结点文本
    :param centerPt: 结点中心坐标
    :param parentPt: 箭头起点坐标
    :param nodeType: 结点类型：decisionNode|leafNode
    :return:
    """
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords="axes fraction",
                            xytext=centerPt,
                            textcoords="axes fraction",
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子结点之间的线条上填充文本信息
    :param cntrPt: 子节点坐标
    :param parentPt: 父节点坐标
    :param txtString: 文本
    :return:
    """
    xMid = (cntrPt[0] + parentPt[0]) / 2
    yMid = (cntrPt[1] + parentPt[1]) / 2
    createPlot.ax1.text(xMid, yMid, txtString)


createPlot(decisionTree)


def storeTree(inputTree, filename):
    """
    将决策树模型序列化到文件，以便下次读取
    :param inputTree: 要序列化的决策树
    :param filename: 保存的文件
    :return:
    """
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


file = "dtmodel"
storeTree(decisionTree, file)


def loadTree(filename):
    """
    加载存储在文件内的决策树模型
    :param filename: 文件路径
    :return: 决策树
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


loadedTree = loadTree(file)
print(loadedTree)


def classify(inputTree, labels, testVec):
    """
    对输入使用决策树进行分类
    :param inputTree: 预先生成的决策树模型
    :param labels: 特征说明
    :param testVec: 需要分类的数据
    :return: 分类的结果
    """
    firstKey = list(inputTree.keys())[0]
    subDict = inputTree[firstKey]
    feature = labels.index(firstKey)
    classLabel = ""
    for key in subDict.keys():
        if testVec[feature] == key:
            if isinstance(subDict[key], dict):
                classLabel = classify(subDict[key], labels, testVec)    # 继续分类
            else:
                classLabel = subDict[key]
    return classLabel


data = ["young", "hyper", "yes", "reduced"]
print(classify(loadedTree, labels, data))





