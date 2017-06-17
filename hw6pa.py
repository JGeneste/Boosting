'''CSE 151 HW6 P3
Jason Geneste A11357496
Rachel Keirouz A11344666
Daniel Keirouz A12948260'''

import numpy
import sys
import math
import random


############################# Function Definitions #############################

'''function that gets the data from the file into a list of tuples 
of the form (featureVector,label)'''
def getData(fileName):
    retSet = []
    with open(fileName) as data:
        for line in data:
            lineL = [int(x) for x in line.split()]
            label = lineL[-1]
            del lineL[-1]
            retSet.append((lineL,label))
    return retSet


'''weak learner. "i" is a word in the dictionary. "x" is an email. "clas"
is 0 or 1, 2 being hi- and 1 being hi+ '''
def weakLearner(i, x, clas):
    if clas == 1:
        if x[i] == 1:
            return 1
        else:
            return -1
    else:
        if x[i] == 1:
            return -1
        else:
            return 1


def boosting(dataSet, t):
    length = len(dataSet)
    d = [1./length]*length
    length2 = 4003
    finalClassifier = []
    for rounds in range(t):
        #var to keep track of what email we're on
        numEmail = 0
        '''these will be 4003 elements long, each element indicating the
        cumulative error of each classifier'''
        learn1Err = [0.0]*length2
        learn2Err = [0.0]*length2
        lowestErrValue = float("inf")
        lowestErrIndex = 0
        lowestErrClass = 0
        temp = 0.0
        learnRet = 0
        for (featVec,label) in dataSet:
            for index in range(length2):
                learnRet = weakLearner(index,featVec,1)
                #if labeled incorrectly, increase error
                if (learnRet != label):
                    temp = learn1Err[index] + d[numEmail]
                    learn1Err[index] = temp
                learnRet = weakLearner(index,featVec,2)
                if (learnRet != label):
                    temp = learn2Err[index] + d[numEmail]
                    learn2Err[index] = temp
            numEmail += 1
        for x in range(length2):
            if(learn1Err[x] < lowestErrValue):
                lowestErrValue = learn1Err[x]
                lowestErrIndex = x
                lowestErrClass = 1
            if(learn2Err[x] < lowestErrValue):
                lowestErrValue = learn2Err[x]
                lowestErrIndex = x
                lowestErrClass = 2
        alpha = (math.log((1 - lowestErrValue) / lowestErrValue))/2
        numEmail = 0
        z = 0.0
        update = []
        for (featVec, label) in dataSet:
            #print lowestErrIndex
            temp = (d[numEmail]*math.exp(-1*(alpha*label*weakLearner(lowestErrIndex,featVec,lowestErrClass))))
            update.append(temp)
            z += temp
            numEmail += 1
        for index in range(length):
            d[index] = update[index] / z
        finalClassifier.append((alpha,lowestErrIndex,lowestErrClass))
    return finalClassifier

def getError(dataset, classifierSet):
    numErrors = 0
    for(line, label) in dataset:
        currSum = 0
        for(alpha, index, clas) in classifierSet:
            currValue = alpha * weakLearner(index, line, clas)
            currSum += currValue
        if currSum == 0:
            rand = random.randint(0, 1)
            if rand == 1:
                prediction = 1
            else:
                prediction = -1
        else:
            prediction = 1 if currSum > 0 else -1
        if(prediction != label):
            numErrors = numErrors + 1
    return float(numErrors) / float(len(dataset))


def getWords(fileName, c):
    retSet = []
    dictSet = []
    '''with open(fileName) as dictionary:
        for line in dictionary:
            dictSet.append(line)'''
    with open(fileName) as dictionary:
        dictSet = [line.rstrip('\n') for line in dictionary]
    for (alpha,index,clas) in c:
        retSet.append(dictSet[index])
    return retSet


################################### Testing ###################################


trainData = getData('hw6train.txt')
testData = getData('hw6test.txt')
rounds = [3,4,7,10,15,20]
#rounds = [10]
for numRounds in rounds:
    classSet = boosting(trainData,numRounds)
    trainError = getError(trainData,classSet)
    testError = getError(testData,classSet)
    print "*********************************************"
    print str(numRounds) + " rounds of boosting"
    print "Training Error: " + str(trainError)
    print "Test Error:     " + str(testError)
    if(numRounds == 10):
        theWords = getWords('hw6dictionary.txt',classSet)
        wordsStr = ', '.join(map(str, theWords))
        print "Chosen Words: " + wordsStr
    print "*********************************************"


