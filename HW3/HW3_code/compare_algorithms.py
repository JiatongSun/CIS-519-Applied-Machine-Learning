import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from cross_validation import cross_validated_accuracy
from logistic_regression import LogisticRegression
from adagrad import LogisticRegressionAdagrad
from preprocessing import preprocessing, convertToBinarySeries

if __name__ == '__main__':
#    filepath = "hw3-wdbc.csv"
#    filepath = "hw3-retinopathy.csv"
    filepath = "hw3-diabetes.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:-1]]
    y = df[df.columns[-1]]

    n,d = X.shape
    
#    X = preprocessing(X,0.5)
    y = convertToBinarySeries(y)
    
    standardizer = StandardScaler()
    X = pd.DataFrame(standardizer.fit_transform(X))
    
    regLambda = 1
    
#    logregModel1 = LogisticRegression(regLambda = regLambda, regNorm=1, alpha = 0.001)
#    logregModel2 = LogisticRegression(regLambda = regLambda, regNorm=2, alpha = 0.001)
#    logregModel3 = LogisticRegressionAdagrad(regLambda = regLambda, regNorm=1, alpha = 0.001)
    logregModel4 = LogisticRegressionAdagrad(regLambda = regLambda, regNorm=2, alpha = 0.001)
    
#    score1 = cross_validated_accuracy(logregModel1,X,y,3,5)
#    score2 = cross_validated_accuracy(logregModel2,X,y,3,5)
#    score3 = cross_validated_accuracy(logregModel3,X,y,3,5)
    score4 = cross_validated_accuracy(logregModel4,X,y,3,5)
    
#    print(f'\nscore1: {score1}')
#    print(f'score2: {score2}')
#    print(f'score3: {score3}')
    print(f'score4: {score4}')
    
    