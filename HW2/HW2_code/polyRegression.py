import pandas as pd

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

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
        self.JHist = None
        self.theta = np.random.randn(degree+1).reshape(-1,1)
        self.alpha = 0.25
        self.thresh = 1E-4
        self.mean = np.zeros((degree,1))
        self.std = np.zeros((degree,1))


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
        X = self.standardize_train(X)
        X = X.to_numpy()
        y = y.to_numpy()
        n = len(y)
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        
        if self.tuneLambda and self.regLambdaValues != []:
            model = Ridge()
            grid = GridSearchCV(estimator = model,cv=3,
                                param_grid = dict(alpha = self.regLambdaValues))
            grid.fit(X,y.reshape(-1,1))
            self.regLambda = grid.best_params_.get('alpha')
            print(f'best lambda: {self.regLambda}')

        self.theta = self.gradientDescent(X,y,self.theta)
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        X = self.polyfeatures(X,self.degree)
        X = self.standardize_test(X)
        X = X.to_numpy()
        n = X.shape[0]
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        
        return pd.DataFrame(np.dot(X,self.theta))
    
    def standardize_train(self, X):
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
    
    def standardize_test(self, X):
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
    
    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** Not returning a matrix with just one value! **
        '''
        n,d = X.shape
        yhat = np.dot(X,theta)
        y = y.reshape(-1,1)
        J = np.dot((yhat-y).T,(yhat-y))/n\
                + self.regLambda*np.sum(theta[1:]**2)
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar
    
    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        y = y.reshape(-1,1)
        self.JHist = []
        iter_num = 0
        last_cost = 1E8
        activate_strict = False
        while True:
            cost = self.computeCost(X, y, theta)
            self.JHist.append( (cost, theta) )
#            if iter_num > 0 and\
#                abs(cost - last_cost) < self.thresh:
#                    print(cost)
#                    break
            if iter_num > 0 and\
                np.linalg.norm(theta - self.JHist[-2][-1]) <= self.thresh:
#                print("Iteration: ", iter_num+1, 
#                  " \nCost: ", self.JHist[iter_num][0],
#                  " \nTheta:\n ", theta)
                print(cost)
                break
#            if iter_num >0:
#                check_1 = np.linalg.norm(theta - self.JHist[-2][-1])
#                check_2 = abs(cost - last_cost)
#                if check_2 < self.thresh and activate_strict is not True:
#                    if iter_num < 100:
#                        activate_strict = True
#                    else:
#                        print(cost)
#                        break
#                if check_1 < self.thresh and activate_strict is True:
#                    print(cost)
#                    break
#                    
#            yhat = np.dot(X,theta)
#            theta = theta - np.dot(X.T, (yhat-y)) * (self.alpha / n)
#            theta[1:] = theta[1:] * (1 - self.alpha * self.regLambda)
            
            for i in range(theta.shape[0]):
                yhat = np.dot(X,theta)
                if i==0:
                    theta[i] = theta[i]-np.dot(X[:,i], (yhat-y))*(self.alpha/n)
                else:
                    theta[i] = theta[i]*(1-self.alpha*self.regLambda)
                    theta[i] = theta[i]-np.dot(X[:,i], (yhat-y))*(self.alpha/n)
            iter_num += 1
            last_cost = cost
        return theta
    
    
def test_polyreg_univariate():
    '''
        Test polynomial regression
    '''

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-polydata.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    # regression with degree = d
    d = 8
    
    # no regularization
    model = PolynomialRegression(degree = d, regLambda = 0)
    model.fit(X, y)
    
    # output predictions
    xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure(1,figsize=(16,9))
    plt.plot(X, y, 'rx')
    plt.title('Regularization Parameter $\lambda$ = 0',fontsize=22)
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X',fontsize=22)
    plt.ylabel('Y',fontsize=22)
    plt.show()
    
    # tune the hyperparameter
    lambda_list = list([0, 0.001, 0.003, 0.006, 0.01,
                       0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 10])
    model = PolynomialRegression(degree = d, regLambda = 0.0001,
                tuneLambda = True, regLambdaValues = lambda_list)
    model.fit(X, y)
    
    # output predictions
    xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure(2,figsize=(16,9))
    plt.plot(X, y, 'rx')
    plt.title('Regularization Parameter $\lambda$ = '+str(model.regLambda)
                ,fontsize=22)
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X',fontsize=22)
    plt.ylabel('Y',fontsize=22)
    plt.show()

if __name__ == '__main__':
    test_polyreg_univariate()