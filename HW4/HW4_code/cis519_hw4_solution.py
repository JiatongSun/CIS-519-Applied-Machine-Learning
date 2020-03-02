import numpy as np
import pandas as pd
import math
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor

        Class Fields 
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''

        self.clfs = []  # keep the class fields, and be sure to keep them updated during boosting
        self.betas = []
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.K = None
        self.classes = None
        
        #TODO



    def fit(self, X, y, random_state=None):
        '''
        Trains the model. 
        Be sure to initialize all individual Decision trees with the provided random_state value if provided.
        
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        #TODO
        X = X.to_numpy()
        y = y.to_numpy()
        n,d = X.shape
        y = y.reshape(-1,1)
        
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        weights = np.full(n,1/n).reshape(-1,1)
        
        for iter_num in range(self.numBoostingIters):
            h = tree.DecisionTreeClassifier(max_depth = self.maxTreeDepth,
                                              random_state = random_state)
            h.fit(X,y,sample_weight = weights.flatten())
            self.clfs.append(h)
            y_pred = h.predict(X).reshape(-1,1)
            epsilon = np.sum((y_pred!=y)*weights)
            beta = np.log((self.K-1)*(1-epsilon)/epsilon)/2
            self.betas.append(beta)
            weights[y_pred==y] *= np.exp(-beta)
            weights[y_pred!=y] *= np.exp(beta)
            weights /= sum(weights)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
        '''
        #TODO
        X = X.to_numpy()
        n,d = X.shape
        proba = np.zeros((n,self.K))
        for iter_num in range(self.numBoostingIters):
            proba += self.clfs[iter_num].predict_proba(X)
        max_proba = np.argmax(proba,axis=1).reshape(-1)
        pred_array = np.tile(self.classes,(n,1))
        y_pred = np.choose(max_proba,pred_array.T).reshape(-1,1)
        return pd.DataFrame(y_pred)