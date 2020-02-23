import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

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
                if self.hasConverged(theta,old_theta,self.epsilon):
                    print(f'cost: {cost}')
                    print(f'iteration: {iter_num}')
                    self.theta = theta.copy()
                    return
                if iter_num > self.maxNumIters:
                    print(f'cost: {cost}')
                    print(f'Reach maximum iterations!')
                    self.theta = theta.copy()
                    return
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
        return pd.DataFrame(self.sigmoid(X*self.theta))



    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1+np.exp(-Z))
    
def mapFeature(X, column1, column2, maxPower = 6):
    '''
    Maps the two specified input features to quadratic features. Does not standardize any features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the maxPower polynomial
        
    Arguments:
        X is an n-by-d Pandas data frame, where d > 2
        column1 is the string specifying the column name corresponding to feature X1
        column2 is the string specifying the column name corresponding to feature X2
    Returns:
        an n-by-d2 Pandas data frame, where each row represents the original features augmented with the new features of the corresponding instance
    '''
    new_X = X.copy()
    for power in range(maxPower+1):
        for i in range(power+1):
            X1, X2 = X[column1], X[column2]
            new_col = (X1**(power-i))*(X2**i)
            new_X = pd.concat([new_X,new_col],axis=1)
    new_X.columns = np.arange(0,new_X.shape[1])
    new_X = new_X.drop([0,1],axis=1)
    return new_X


def test_logreg1():
    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data1.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape

    # # Standardize features
    standardizer = StandardScaler()
    Xstandardized = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    logregModel = LogisticRegressionAdagrad(regLambda = 0.00000001, regNorm = 1)
    logregModel.fit(Xstandardized,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(12, 9))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.title('L2 Regularization, $\lambda$ = 0',fontsize=22)
    plt.xlabel('Exam 1 Score',fontsize=22)
    plt.ylabel('Exam 2 Score',fontsize=22)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
def test_logreg2():

    polyPower = 6

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data2.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape

    # map features into a higher dimensional feature space
    Xaug = mapFeature(X.copy(), X.columns[0], X.columns[1], polyPower)

    # # Standardize features
    standardizer = StandardScaler()
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    logregModel = LogisticRegressionAdagrad(regLambda = 0.00000001, regNorm=2)
    logregModel.fit(Xaug,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = mapFeature(allPoints, allPoints.columns[0], allPoints.columns[1], polyPower)
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # standardize data
    
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()


    print(str(Z.min()) + " " + str(Z.max()))

if __name__ == '__main__':
#    test_logreg1()
    test_logreg2()