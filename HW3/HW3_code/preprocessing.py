import pandas as pd
import numpy as np
import random 

def extractFeatures(X,y,num_feature):
    completeDf = pd.concat([X,y], axis = 1)
    corrDf = completeDf.corr().iloc[-1,:-1]
    corrDf = abs(corrDf).sort_values(ascending=False)
    feature_index = corrDf.index[:num_feature]
    X_out = X[feature_index]
    return X_out, feature_index

def randomFeatures(X,num_feature,random_seed):
    random.seed(random_seed)
    idx = np.arange(0,X.shape[1])
    random.shuffle(idx)
    X_out = X.iloc[:,idx[:10]]
    feature_index = X_out.columns
    return X_out, feature_index
    

def preprocessing(inputDf,missing_thresh):
    outputDf = inputDf.copy()
    # drop feature that misses too many data
    outputDf = dropMissingColumn(outputDf,missing_thresh)
    # replenish missing data
    outputDf = fillMissingColumn(outputDf)
    return outputDf

def dropMissingColumn(inputDf,missing_thresh):
    outputDf = inputDf.copy()
    missDf = getMissingRatio(outputDf)
    missIdx = missDf['MissingPercent'] >= missing_thresh
    missFeat = missDf.loc[missIdx,'Feature']
    outputDf = outputDf.drop(missFeat,axis=1)
    return outputDf

def fillMissingColumn(inputDf):
    outputDf = inputDf.copy()
    # replenish numerical with mean
    numerical_col = (outputDf.iloc[:,].dtypes!='O').values
    numericalDf = outputDf.iloc[:,numerical_col]
    numericalDf = numericalDf.fillna(numericalDf.mean())
    # replenish category with mode
    category_col = (outputDf.iloc[:,].dtypes=='O').values
    categoryDf = outputDf.iloc[:,category_col]
    categoryDf = categoryDf.fillna(categoryDf.mode().iloc[0])
    if not categoryDf.empty:
        categoryDf = pd.get_dummies(categoryDf)
    # combine two dataframes
    outputDf = pd.concat([numericalDf,categoryDf],axis=1)
    return outputDf

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

def convertToBinarySeries(inputSeries):
    if inputSeries.isna().sum() > 0:
        raise ValueError('cannot convert')
        return
    firstVal = inputSeries.loc[0]
    for i in range(len(inputSeries)):
        if inputSeries.loc[i] != firstVal:
            secondVal = inputSeries.loc[i]
            break
    outSeries = inputSeries.copy()
    outSeries[inputSeries == firstVal] = 1
    outSeries[inputSeries == secondVal] = 0
    return outSeries

def addDummyFeatures(inputDf, feature):
    if feature not in inputDf.columns:
        raise ValueError('not a feature')
        return
    outDf =  (pd.concat([inputDf,pd.get_dummies(inputDf[feature],\
                        prefix=feature)],axis=1)).drop(feature,axis=1)
    return outDf


