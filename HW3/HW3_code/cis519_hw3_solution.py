import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2,
                 epsilon=0.0001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
            alpha is the learning rate
            regLambda is the regularization parameter
            regNorm is the type of regularization 
            (either L1 or L2, denoted by a 1 or a 2)
            epsilon is the convergence parameter
            maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.initTheta = initTheta
        self.finalTheta = None
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  
            ** make certain you're not returning a 1 x 1 matrix! **
        '''
        h = self.sigmoid(X*theta) # h is a n-by-1 numpy matrix
        cost_1 = -y.T*np.log(h)-(1-y.T)*np.log(1-h)
        if self.regNorm==2:
            cost_2 = regLambda*theta[1:].T*theta[1:]
        elif self.regNorm==1:
            cost_2 = regLambda*np.abs(theta[1:]).sum()
        else:
            raise ValueError('regNorm is not 1 or 2!')
        cost = cost_1 + cost_2
        cost = cost.item() # convert 1 x 1 matrix to scalar
        return cost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        h = self.sigmoid(X*theta) # h is a n-by-1 numpy matrix
        grad_1 = X.T*(h-y)
        if self.regNorm==2:
            grad_2 = regLambda*theta
        elif self.regNorm==1:
            grad_2 = regLambda*np.sign(theta)
        else:
            raise ValueError('regNorm is not 1 or 2!')
        grad_2[0] = 0
        grad = grad_1 + grad_2 # grad is a d-by-1 numpy matrix
        return grad
    
    def hasConverged(self, new_theta, old_theta, epsilon):
        '''
        Determine if learning process has converged
        Arguments:
            new_theta is a d-by-1 numpy matrix
            old_theta is a d-by-1 numpy matrix
            epsilon is convergence threshold
        Returns:
            boolean variable representing convergence
        '''
        return np.linalg.norm(new_theta-old_theta)<epsilon


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        n,d = X.shape
        if self.initTheta is None:
            self.initTheta = np.zeros((d+1,1))
        self.initTheta = np.asmatrix(self.initTheta)
        # initialize
        X = np.c_[np.ones((n,1)), X]
        theta = self.initTheta.copy()
        old_theta = self.initTheta.copy()
        iter_num = 0
        # change data frame to numpy matrix
        X = np.asmatrix(X)
        y = np.asmatrix(y.to_numpy())
        y = y.reshape(-1,1)
        # gradient descent
        while True:
            # compute cost
            cost = self.computeCost(theta, X, y, self.regLambda)
            grad = self.computeGradient(theta, X, y, self.regLambda)
            theta -= self.alpha*grad
            if self.hasConverged(theta,old_theta,self.epsilon):
                print(f'cost: {cost}')
                print(f'iteration: {iter_num}')
                self.finalTheta = theta.copy()
                break
            if iter_num > self.maxNumIters:
                print(f'cost: {cost}')
                print(f'Reach maximum iterations!')
                self.finalTheta = theta.copy()
                break
            old_theta = theta.copy()
            iter_num += 1


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        return pd.DataFrame(self.predict_proba(X)>0.5).astype(int)

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''
        n,_ = X.shape
        X = np.c_[np.ones((n,1)), X]
        X = np.asmatrix(X)
        return pd.DataFrame(self.sigmoid(X*self.finalTheta))



    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1+np.exp(-Z))


class LogisticRegressionAdagrad:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2,
                 epsilon=0.0001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
            alpha is the learning rate
            regLambda is the regularization parameter
            regNorm is the type of regularization 
            (either L1 or L2, denoted by a 1 or a 2)
            epsilon is the convergence parameter
            maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.initTheta = initTheta
        self.theta = None
        self.cumulateGrad = None
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  
            ** make certain you're not returning a 1 x 1 matrix! **
        '''
        h = self.sigmoid(X*theta) # h is a n-by-1 numpy matrix
        cost_1 = -y.T*np.log(h)-(1-y.T)*np.log(1-h)
        if self.regNorm==2:
            cost_2 = regLambda*theta[1:].T*theta[1:]
        elif self.regNorm==1:
            cost_2 = regLambda*np.abs(theta[1:]).sum()
        else:
            raise ValueError('regNorm is not 1 or 2!')
        cost = cost_1 + cost_2
        cost = cost.item() # convert 1 x 1 matrix to scalar
        return cost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient according to Adagrad
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
        Returns:
            the gradient, an d-dimensional vector
        '''
        zeta = 1E-7
        h = self.sigmoid(X*theta)
        gk_1 = X.T*(h-y)
        if self.regNorm==2:
            gk_2 = regLambda*theta
        elif self.regNorm==1:
            gk_2 = regLambda*np.sign(theta)
        else:
            raise ValueError('regNorm is not 1 or 2!')
        gk_2[0] = 0
        gk = gk_1 + gk_2
        self.cumulateGrad += np.multiply(gk,gk)
        grad = gk/(np.sqrt(self.cumulateGrad)+zeta)
        return grad
    
    
    def hasConverged(self, new_theta, old_theta, epsilon):
        '''
        Determine if learning process has converged
        Arguments:
            new_theta is a d-by-1 numpy matrix
            old_theta is a d-by-1 numpy matrix
            epsilon is convergence threshold
        Returns:
            boolean variable representing convergence
        '''
        return np.linalg.norm(new_theta-old_theta)<epsilon


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        n,d = X.shape
        if self.initTheta is None:
            self.initTheta = np.zeros((d+1,1))
        self.initTheta = np.asmatrix(self.initTheta)
        # initialize
        X = np.c_[np.ones((n,1)), X]
        theta = self.initTheta.copy()
        old_theta = self.initTheta.copy()
        iter_num = 0
        self.cumulateGrad = np.asmatrix(np.zeros((d+1,1)))
        # change data frame to numpy matrix
        X = np.asmatrix(X)
        y = np.asmatrix(y.to_numpy())
        y = y.reshape(-1,1)
        # Stochastic Gradient Descent
        while True:
            # compute cost
            cost = self.computeCost(theta, X, y, self.regLambda)
            # shuffle data set
            X,y = shuffle(X,y)
            for instance in range(n):
                cur_X, cur_y = X[instance], y[instance]
                grad = self.computeGradient(theta, cur_X, cur_y, self.regLambda)
                theta -= self.alpha*grad
                
                if iter_num > self.maxNumIters:
                    print(f'cost: {cost}')
                    print(f'Reach maximum iterations!')
                    self.theta = theta.copy()
                    return
                
            if self.hasConverged(theta,old_theta,self.epsilon):
                    print(f'cost: {cost}')
                    print(f'iteration: {iter_num}')
                    self.theta = theta.copy()
                    return
            old_theta = theta.copy()
            print(cost)
            iter_num += 1
                


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        return pd.DataFrame(self.predict_proba(X)>0.5).astype(int)

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''
        n,_ = X.shape
        X = np.c_[np.ones((n,1)), X]
        X = np.asmatrix(X)
        return pd.DataFrame(self.sigmoid(X*self.theta))


    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1+np.exp(-Z))