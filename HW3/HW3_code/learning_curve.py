import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from cross_validation import cross_validated_accuracy
from logistic_regression import LogisticRegression
from adagrad import LogisticRegressionAdagrad
from preprocessing import preprocessing, convertToBinarySeries

def learning_curves(model, X, y, max_iteration):
    accuracy_list = []
    num_trials = 5
    num_folds = 5
    for iter_num in range(max_iteration):
        model.maxNumIters = iter_num
        cvScore = cross_validated_accuracy(model, X, y, num_trials, num_folds)
        accuracy_list.append(cvScore)
        print(f'iter_num: {iter_num}')
    return accuracy_list

if __name__ == '__main__':
    filepath = "hw3-wdbc.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:-1]]
    y = df[df.columns[-1]]

    n,d = X.shape
    
    y = convertToBinarySeries(y)
    
    standardizer = StandardScaler()
    X = pd.DataFrame(standardizer.fit_transform(X))
    
    logregModel = LogisticRegression(regLambda = 1, regNorm=1, alpha = 0.001)
#    logregModel = LogisticRegression(regLambda = 1, regNorm=1, alpha = 0.01)
#    logregModel = LogisticRegression(regLambda = 1, regNorm=1, alpha = 0.1)
    
#    logregModel = LogisticRegression(regLambda = 1, regNorm=2, alpha = 0.001)
#    logregModel = LogisticRegression(regLambda = 1, regNorm=2, alpha = 0.01)
#    logregModel = LogisticRegression(regLambda = 1, regNorm=2, alpha = 0.1)
    
#    logregModel = LogisticRegressionAdagrad(regLambda = 0, regNorm=1, alpha = 0.01)
#    logregModel = LogisticRegressionAdagrad(regLambda = 1, regNorm=1, alpha = 0.01)
    
#    logregModel = LogisticRegressionAdagrad(regLambda = 1, regNorm=2, alpha = 0.001)

    max_iteration = 50
    
    accuracy_list = learning_curves(logregModel, X, y, max_iteration)
    
    plt.figure(1,figsize = (8,6))
    plt.plot(np.arange(max_iteration),accuracy_list)
    plt.tick_params(labelsize = 15)
    plt.title('Adagrad, L1 Norm, $\lambda = 1$',fontsize = 24)
    plt.xlabel('Iteration',fontsize = 20)
    plt.ylabel('Mean Accuracy',fontsize = 20)
    plt.grid(b=True)
    plt.show()