import pandas as pd

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8
                 , tuneLambda = False, regLambdaValues = []):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.tuneLambda = tuneLambda
        self.regLambdaValues = regLambdaValues
        self.theta = np.zeros(degree+1).reshape(-1,1)
        
        # autograder parameter
        self.alpha = 0.25
        self.thresh = 1E-4
        
#        # self program parameter
#        self.alpha = 0.1
#        self.thresh = 1E-2
        
        self.mean = np.zeros((degree,1))
        self.std = np.zeros((degree,1))
        self.is_tuning = False
        self.tune_iter = 500000
        self.kfold = 5
        self.trial = 5


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d data frame, with each column comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 data frame
            degree is a positive integer
        '''
        base = X.to_numpy()
        for i in range(degree):
            if i == 0:
                poly_feat = base
            else:
                poly_feat = np.c_[poly_feat,base**(i+1)]
        return pd.DataFrame(poly_feat)
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 data frame
                y is an n-by-1 data frame
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling first
        '''
        X = self.polyfeatures(X,self.degree)
        X = self.standardizeTrain(X)
        X = X.to_numpy()
        y = y.to_numpy()
        n = len(y)
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        
        if self.tuneLambda and self.regLambdaValues != []:
            self.is_tuning = True
            
            # use sklearn Ridge
            model = Ridge()
            grid = GridSearchCV(estimator = model,
                                param_grid = dict(alpha = self.regLambdaValues))
            grid.fit(X,y.reshape(-1,1))
            self.regLambda = grid.best_params_.get('alpha')
            
#            # use self programmed cross validation
#            self.regLambda = self.findLambda(X,y.reshape(-1,1),self.theta)
            
            print(f'best lambda: {self.regLambda}')
            self.is_tuning = False

        self.theta, cost = self.gradientDescent(X,y,self.theta,self.regLambda)
        print(f'cost: {cost}')
        
    def findLambda(self, X, y, theta):
        '''
        Find the best regularization lambda
        Arguments:
          X is a n-by-d numpy array
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value representing the best lambda
        '''
        min_cost = 1E8
        best_lambda = 0
        for regLambda in self.regLambdaValues:
            cur_cost = self.regPerformance(X, y, theta, regLambda)
            print(f'lambda: {regLambda}, cost:{cur_cost}')
            if cur_cost < min_cost:
                min_cost = cur_cost
                best_lambda = regLambda
        return best_lambda
    
    def regPerformance(self, X, y, theta, regLambda):
        '''
        Calculate the coefficient of the prediction
        Arguments:
          X is a n-by-d numpy array
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value representing the coefficient
        '''
#        total_coef = 0
        total_cost = 0
        y = y.reshape(-1)
        
        rkf = RepeatedKFold(n_splits=self.kfold,
                            n_repeats=self.trial, random_state=0)
        for train_index, test_index in rkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            cur_theta, _ = self.gradientDescent(X_train,
                                    y_train, theta, regLambda)
            yhat = np.dot(X_test, cur_theta).reshape(-1)
            
            # criterion 1: cost
            cost = np.linalg.norm(yhat - y_test)/len(y_test)
            total_cost += cost

#            # criterion 2: coefficience
#            u = np.sum((yhat - y_test)**2)
#            v = np.sum((y_test-y_test.mean())**2)
#            coef = 1-u/v
#            total_coef += coef
            
        return (total_cost)
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        X = self.polyfeatures(X,self.degree)
        X = self.standardizeTest(X)
        X = X.to_numpy()
        n = X.shape[0]
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        
        return pd.DataFrame(np.dot(X,self.theta))
    
    def standardizeTrain(self, X):
        '''
        standardize the training data before training or predicting
        Arguments:
            X is a n-by-d data frame
        Returns:
            an n-by-d data frame of the predictions
        '''
        X = X.to_numpy()
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        standard = (X - self.mean) / self.std
        return pd.DataFrame(standard)
    
    def standardizeTest(self, X):
        '''
        standardize the test data before training or predicting
        Arguments:
            X is a n-by-d data frame
        Returns:
            an n-by-d data frame of the predictions
        '''
        X = X.to_numpy()
        standard = (X - self.mean) / self.std
        return pd.DataFrame(standard)
    
    def computeCost(self, X, y, theta, regLambda):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy array
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
          regLambda is a scalar
        Returns:
          a scalar value of the cost  
              ** Not returning a matrix with just one value! **
        '''
        n,d = X.shape
        yhat = np.dot(X,theta)
        y = y.reshape(-1,1)
        J = np.dot((yhat-y).T,(yhat-y))/n + regLambda * np.sum(theta[1:]**2)
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar
    
    def gradientDescent(self, X, y, theta, regLambda):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
            regLambda is a scalar
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        y = y.reshape(-1,1)
        iter_num = 0
        last_cost = 1E8
        last_theta = self.theta
        while True:
            # compute cost
            cost = self.computeCost(X, y, theta, regLambda)
            
            # gradient descent
            yhat = np.dot(X,theta)
            theta = theta - np.dot(X.T, (yhat-y)) * (self.alpha / n)
            theta[1:] = theta[1:] * (1 - self.alpha * regLambda)
            
            # judge convergence
            L2_criterion = np.linalg.norm(theta - last_theta)
            cost_criterion = abs(cost - last_cost)
            
#            if L2_criterion <= self.thresh:
            if cost_criterion < self.thresh:
                break
            
            if self.is_tuning is True and iter_num > self.tune_iter:
                break
            
            # update
            iter_num += 1
            last_theta = theta
            last_cost = cost
        return theta, cost