import numpy
import re
import operator
import feedparser


def loadDataSet():
    """
    创建待训练数据集
    :return:
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]  # 训练文档集合
    classVec = [0, 1, 0, 1, 0, 1]  # 对应每篇文章的分类
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建词汇表
    :param dataSet: 输入训练文档集
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    """
    根据输入的词汇表和文档，判断文档中是否含有词汇表中各个单词
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # returnVec[vocabList.index(word)] = 1    # 词集模型
            returnVec[vocabList.index(word)] += 1  # 词袋模型
        else:
            # print("the word: %s is not in my Vocabulary!" % word)
            continue
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # p0Num = numpy.zeros(numWords)
    # p1Num = numpy.zeros(numWords)
    # p0Denom = 0
    # p1Denom = 0

    p0Num = numpy.ones(numWords)
    p1Num = numpy.ones(numWords)
    p0Denom = 2
    p1Denom = 2

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = numpy.log(p0Num / p0Denom)
    p1Vect = numpy.log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    分类函数
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB(testEntry):
    """

    :param testEntry:
    :return:
    """
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNB0(numpy.array(trainMat), numpy.array(listClasses))
    print(p0V)
    print(p1V)
    print(pAb)
    thisDoc = numpy.array(setOfWord2Vec(myVocabList, testEntry))
    thisType = classifyNB(thisDoc, p0V, p1V, pAb)
    print(testEntry, 'classified as: ', thisType)


# doc = ['love', 'my', 'dalmation']
doc = ['stupid', 'garbage']
testingNB(doc)


def textParse(bigString):
    """
    解析文本，返回小写字母的单词数组
    :param bigString:
    :return:
    """
    listOfTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 2]


def spamText():
    """

    :return:
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(numpy.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        classType = classifyNB(numpy.array(wordVector), p0V, p1V, pSpam)
        if classType != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount) / len(testSet)
    print('the error rate is: ', errorRate)


spamText()


def calMostFreq(vocabList, fullText):
    """

    :param vocabList:
    :param fullText:
    :return:
    """
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """

    :param feed1:
    :param feed0:
    :return:
    """
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(10):
        randIndex = int(numpy.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        classType = classifyNB(numpy.array(wordVector), p0V, p1V, pSpam)
        if classType != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(shanghai, beijing):
    """

    :param shanghai:
    :param beijing:
    :return:
    """
    vocabList, p0V, p1V = localWords(shanghai, beijing)
    topShanghai = []
    topBeijing = []
    for i in range(len(p0V)):
        if p0V[i] > -5.0:
            topShanghai.append((vocabList[i], p0V[i]))
        if p1V[i] > -5.0:
            topBeijing.append((vocabList[i], p1V[i]))
    sortedShanghai = sorted(topShanghai, key=lambda pair: pair[1], reverse=True)
    print("------------------------SHANGHAI--------------------------")
    for item in sortedShanghai:
        print(item[0])
    sortedBeijing = sorted(topBeijing, key=lambda pair: pair[1], reverse=True)
    print("------------------------BEIJING--------------------------")
    for item in sortedBeijing:
        print(item[0])


shanghai = feedparser.parse('https://shanghai.craigslist.org/search/jjj?format=rss')
beijing = feedparser.parse('https://beijing.craigslist.org/search/jjj?format=rss')
getTopWords(shanghai, beijing)
