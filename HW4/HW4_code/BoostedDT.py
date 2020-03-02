import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from preprocessing import sortData, dropMissingColumn
from preprocessing import fillMissingColumnSpecify, filterUnlabel


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
        
        
        
def test_boostedDT():
    # load the data set
    sklearn_dataset = datasets.load_breast_cancer()
    # convert to pandas df
    df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
    df['CLASS'] = pd.Series(sklearn_dataset.target)
    df.head()

    # split randomly into training/testing
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    # Split into X,y matrices
    X_train = train.drop(['CLASS'], axis=1)
    y_train = train['CLASS']
    X_test = test.drop(['CLASS'], axis=1)
    y_test = test['CLASS']


    # train the decision tree
    modelDT = DecisionTreeClassifier()
    modelDT.fit(X_train, y_train)

    # train the boosted DT
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
    modelBoostedDT.fit(X_train, y_train)

    # train sklearn's implementation of Adaboost
    modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
    modelSKBoostedDT.fit(X_train, y_train)

    # output predictions on the test data
    ypred_DT = modelDT.predict(X_test)
    ypred_BoostedDT = modelBoostedDT.predict(X_test)
    ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)

    # compute the training accuracy of the model
    accuracy_DT = accuracy_score(y_test, ypred_DT)
    accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
    accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

    print("Decision Tree Accuracy = "+str(accuracy_DT))
    print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
    print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
    print()
    print("Note that due to randomization, your boostedDT might not always have the ")
    print("exact same accuracy as Sklearn's boostedDT.  But, on repeated runs, they ")
    print("should be roughly equivalent and should usually exceed the standard DT.")

def train_chocolate():
    X = pd.read_csv('ChocolatePipes_trainData.csv')
    y = pd.read_csv('ChocolatePipes_trainLabels.csv')
    X, y = sortData(X, y, 'id')
    useless_feature = ['id', 'Date of entry', 'Country funded by', 
                       'oompa loomper', 'Region code', 'District code',
                       'Chocolate consumers in town', 
                       'Does factory offer tours', 'Recorded by',
                       'Oompa loompa management', 'Payment scheme',
                       'management_group']
#    useless_feature = ['id', 'Date of entry', 'Recorded by']
    X = X.drop(useless_feature,axis=1)
    
    X = dropMissingColumn(X, 0.5)
    categorical_feature = ['chocolate_quality', 'chocolate_quantity',
                           'pipe_type', 'chocolate_source',
                           'chocolate_source_class', 'Cocoa farm',
                           'Official or Unofficial pipe', 
                           'Type of pump','management']
#    categorical_feature = ['chocolate_quality', 'chocolate_quantity',
#                           'pipe_type', 'chocolate_source',
#                           'Official or Unofficial pipe',
#                           'chocolate_source_class', 'Cocoa farm',
#                           'Type of pump','management', 'Country funded by',
#                           'Region code', 'District code', 'management_group',
#                           'Oompa loompa management', 'Payment scheme',]
    
    
    Xc = X[categorical_feature]
    Xn = X.drop(categorical_feature,axis=1)
    
    X = fillMissingColumnSpecify(Xn, Xc)
    y = y.drop(['id'],axis=1)
    
    X_grade = pd.read_csv('ChocolatePipes_gradingTestData.csv')
    X_grade_id = X_grade['id']
    X_grade = X_grade.drop(useless_feature,axis=1)
    Xc_grade = X_grade[categorical_feature]
    Xn_grade = X_grade.drop(categorical_feature,axis=1)
    X_grade = fillMissingColumnSpecify(Xn_grade, Xc_grade)
    X_grade = filterUnlabel(X, X_grade)
    
    X_leader = pd.read_csv('ChocolatePipes_leaderboardTestData.csv')
    X_leader_id = X_leader['id']
    X_leader = X_leader.drop(useless_feature,axis=1)
    Xc_leader = X_leader[categorical_feature]
    Xn_leader = X_leader.drop(categorical_feature,axis=1)
    X_leader = fillMissingColumnSpecify(Xn_leader, Xc_leader)
    X_leader = filterUnlabel(X, X_leader)
    
    # BoostedDT
    df = pd.concat([X,y],axis=1)
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    X_train = train.drop(['label'], axis=1)
    y_train = train['label']
    X_test = test.drop(['label'], axis=1)
    y_test = test['label']
    
# =============================================================================
#     # tuning the best numBoostingIters and maxTreeDepth
#     max_train_accuracy = 0
#     max_test_accuracy = 0
#     max_train_iter = 0
#     max_test_iter = 0
#     max_train_depth = 0
#     max_test_depth = 0
#     for iter_num in range(100, 200, 10):
#         for depth in range(18,19):
#             modelBoostedDT = BoostedDT(numBoostingIters=iter_num, maxTreeDepth=depth)
#             modelBoostedDT.fit(X_train,y_train)
#             train_accuracy = (modelBoostedDT.predict(X_train).values.reshape(-1)
#                         ==y_train.values).sum()/len(y_train)
#             test_accuracy = (modelBoostedDT.predict(X_test).values.reshape(-1)
#                         ==y_test.values).sum()/len(y_test)
#             print(f'iteration: {iter_num}')
#             print(f'depth: {depth}')
#             print(f'train: {train_accuracy}')
#             print(f'test: {test_accuracy}\n')
#             if train_accuracy > max_train_accuracy:
#                 max_train_accuracy = train_accuracy
#                 max_train_iter = iter_num
#                 max_train_depth = depth
#             if test_accuracy > max_test_accuracy:
#                 max_test_accuracy = test_accuracy
#                 max_test_iter = iter_num
#                 max_test_depth = depth
#     print(f'best train performance: {max_train_accuracy}')
#     print(f'    iteration: {max_train_iter}')
#     print(f'    depth: {max_train_depth}')
#     print(f'best test performance: {max_test_accuracy}')
#     print(f'    iteration: {max_test_iter}')
#     print(f'    depth: {max_test_depth}')
#     modelBoostedDT = BoostedDT(numBoostingIters=max_test_iter, maxTreeDepth=max_test_depth)
# =============================================================================


    # Boosted Decision Tree
    modelBoostedDT = BoostedDT(numBoostingIters=28, maxTreeDepth=18)
    modelBoostedDT.fit(X_train,y_train)
    train_accuracy_boostedDT = (modelBoostedDT.predict(X_train).values.reshape(-1)
                     ==y_train.values).sum()/len(y_train)
    test_accuracy_boostedDT = (modelBoostedDT.predict(X_test).values.reshape(-1)
                     ==y_test.values).sum()/len(y_test)
    print(f'train_accuracy_boostedDT: {train_accuracy_boostedDT}')
    print(f'test_accuracy_boostedDT: {test_accuracy_boostedDT}')
    
    modelBoostedDT.fit(X,y)
    
    y_grade_boost = modelBoostedDT.predict(X_grade)
    output_grade_boost = pd.concat([X_grade_id, y_grade_boost], axis=1)
    output_grade_boost.columns = ['id', 'label']
    output_grade_boost.to_csv('predictions-grading-BoostedDT.csv',index=False)
    
    y_leader_boost = modelBoostedDT.predict(X_leader)
    output_leader_boost = pd.concat([X_leader_id, y_leader_boost], axis=1)
    output_leader_boost.columns = ['id', 'label']
    output_leader_boost.to_csv('predictions-leaderboard-BoostedDT.csv',index=False)


# =============================================================================
#     # Preprocessing for SVM and Logistic Regression
#     X = X.to_numpy()
#     y = y.to_numpy().flatten()
#     
#     X_train = X_train.to_numpy()
#     y_train = y_train.to_numpy().flatten()
#     X_test = X_test.to_numpy()
#     y_test = y_test.to_numpy().flatten()
#     
#     X_grade = X_grade.to_numpy()
#     X_leader = X_leader.to_numpy()
#     
#     standardizer = StandardScaler()
#     Xstandardized = pd.DataFrame(standardizer.fit_transform(X))
#     Xstandardized_train = pd.DataFrame(standardizer.fit_transform(X_train))
#     Xstandardized_test = pd.DataFrame(standardizer.fit_transform(X_test))
#     Xstandardized_grade = pd.DataFrame(standardizer.fit_transform(X_grade))
#     Xstandardized_leader = pd.DataFrame(standardizer.fit_transform(X_leader))
# =============================================================================
    
# =============================================================================
#     # SVM
#     svm_clf = SVC(gamma='auto')
#     svm_clf.fit(Xstandardized_train, y_train)
#     train_accuracy_svm = (svm_clf.predict(Xstandardized_train)
#                             ==y_train).sum()/len(y_train)
#     test_accuracy_svm = (svm_clf.predict(Xstandardized_test)
#                             ==y_test).sum()/len(y_test)
#     print(f'train_accuracy_svm: {train_accuracy_svm}')
#     print(f'test_accuracy_svm: {test_accuracy_svm}')
#     
#     svm_clf.fit(Xstandardized, y)
#     
#     y_grade_svm = pd.DataFrame(svm_clf.predict(Xstandardized_grade))
#     output_grade_svm = pd.concat([X_grade_id, y_grade_svm], axis=1)
#     output_grade_svm.columns = ['id', 'label']
#     output_grade_svm.to_csv('predictions-grading-SVC.csv',index=False)
#     
#     y_leader_svm = pd.DataFrame(svm_clf.predict(Xstandardized_leader))
#     output_leader_svm = pd.concat([X_leader_id, y_leader_svm], axis=1)
#     output_leader_svm.columns = ['id', 'label']
#     output_leader_svm.to_csv('predictions-leaderboard-SVC.csv',index=False)
# =============================================================================
    
# =============================================================================
#     # Logistic Regression
#     logistic_clf = LogisticRegression(random_state=42,max_iter=120)
#     logistic_clf.fit(Xstandardized_train, y_train)
#     train_accuracy_logistic = (logistic_clf.predict(Xstandardized_train)
#                                     ==y_train).sum()/len(y_train)
#     test_accuracy_logistic = (logistic_clf.predict(Xstandardized_test)
#                                     ==y_test).sum()/len(y_test)
#     print(f'train_accuracy_logistic: {train_accuracy_logistic}')
#     print(f'test_accuracy_logistic: {test_accuracy_logistic}')
#     
#     logistic_clf.fit(Xstandardized, y)
#     
#     y_grade_logistic = pd.DataFrame(logistic_clf.predict(Xstandardized_grade))
#     output_grade_logistic = pd.concat([X_grade_id, y_grade_logistic], axis=1)
#     output_grade_logistic.columns = ['id', 'label']
#     output_grade_logistic.to_csv('predictions-grading-logistic.csv',index=False)
#     
#     y_leader_logistic = pd.DataFrame(logistic_clf.predict(Xstandardized_leader))
#     output_leader_logistic = pd.concat([X_leader_id, y_leader_logistic], axis=1)
#     output_leader_logistic.columns = ['id', 'label']
#     output_leader_logistic.to_csv('predictions-leaderboard-logistic.csv',index=False)
# =============================================================================
    
    
if __name__ == '__main__':
#    test_boostedDT()
    train_chocolate()