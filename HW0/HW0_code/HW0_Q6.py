import pandas as pd
import numpy as np


def getMissingRatio(inputDf):
    outSeries = inputDf.isna().mean()                 # calculate missing ratios
    outDf = pd.DataFrame({'Feature':outSeries.index,\
                          'MissingPercent':outSeries.values})
    return outDf

def convertToBinary(inputDf, feature):
    if feature not in inputDf.columns:
        raise ValueError('not a feature')
        return
    inputCol = inputDf[feature]
    if inputCol.isna().sum() > 0:
        raise ValueError('cannot convert')
        return
    firstVal = inputDf.loc[0,feature]
    for i in range(len(inputCol)):
        if inputCol.loc[i] != firstVal:
            secondVal = inputCol.loc[i]
            break
    outDf=pd.DataFrame(index=inputDf[feature].index,columns=[feature])
    outDf[inputCol == firstVal] = 1
    outDf[inputCol == secondVal] = 0
    if outDf[feature].isna().sum() > 0:
        raise ValueError('not a binary feature')
        return
    return outDf

def addDummyFeatures(inputDf, feature):
    if feature not in inputDf.columns:
        raise ValueError('not a feature')
        return
    outDf =  (pd.concat([inputDf,pd.get_dummies(inputDf[feature],\
                        prefix=feature)],axis=1)).drop(feature,axis=1)
    return outDf

if __name__ == '__main__':
    inputDf = pd.read_csv('train.csv')
    missingDf = getMissingRatio(inputDf)
    binaryDf = convertToBinary(inputDf, 'Sex')
    dummyDf = addDummyFeatures(inputDf,'Embarked')