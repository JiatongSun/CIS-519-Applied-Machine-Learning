import random 
import numpy as np
import pandas as pd
import timeit
import graphviz
from sklearn import tree

from preprocessing import preprocessing
from preprocessing import extractFeatures
from preprocessing import randomFeatures

def cross_validated_accuracy(DecisionTreeClassifier, 
                             X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    
    num_correct = 0
    num_test = 0
    accuracy_list = list()
    for i in range(num_trials):
        idx = np.arange(0,X.shape[0])
        random.shuffle(idx)
        X_shuf = X.set_index(idx).sort_index()
        y_shuf = y.copy()
        y_shuf.index = idx
        y_shuf = y_shuf.sort_index()
        for j in range(num_folds):
            X_split = np.array_split(X_shuf, num_folds)
            y_split = np.array_split(y_shuf, num_folds)
            X_test = X_split.pop(j)
            y_test = y_split.pop(j)
            X_train = pd.concat(X_split)
            y_train = pd.concat(y_split)
            model = DecisionTreeClassifier.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            num_correct += (y_pred == y_test).sum()
            num_test += len(y_test)
            accuracy_list.append((y_pred == y_test).sum()/len(y_test))
    cvScore = num_correct / num_test
    x_bar = np.mean(accuracy_list)
    ssd = np.std(accuracy_list)
    num_obs = len(accuracy_list)
    t_param = 2.626   # look into the t-table
    low_bound = round(x_bar - t_param * ssd / np.sqrt(num_obs),3)
    upper_bound = round(x_bar + t_param * ssd / np.sqrt(num_obs),3)
    interval = (low_bound, upper_bound)
    print(f'Confidence Interval: {interval}')
    
    return cvScore

def automatic_dt_pruning(DecisionTreeClassifier,
                         X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    
    ccp_idx = np.linspace(0,1,101)
    last_accuracy = 0
    for cur_ccp in ccp_idx:
        DecisionTreeClassifier.ccp_alpha = cur_ccp
        cur_accuracy = cross_validated_accuracy(DecisionTreeClassifier,
                            X, y, num_trials, num_folds, random_seed)
        print(f'ccp_alpha = {cur_ccp:.2f}: {round(cur_accuracy,3)}\n')
        if last_accuracy - cur_accuracy > 0.01:
            return (cur_ccp - 0.01)
        else:
            last_accuracy = cur_accuracy
    ccp_alpha = cur_ccp
    
    return ccp_alpha

if __name__ == '__main__':
    start = timeit.default_timer()
    df = pd.read_csv('NHANES-diabetes-hw-train.csv')
    random_seed = 42
    num_trials = 10
    num_folds = 10
    T = tree.DecisionTreeClassifier(criterion='entropy')
    
    X = df.drop(['SEQN','DIQ010','DIABETIC'],axis=1)
    y = df['DIABETIC']
    
    print(f'Diabetic Ratio (Label): {y.values.sum()/len(y):.2f}\n')
    
    X = preprocessing(X,0.6)
    
#    cvScore = cross_validated_accuracy(T, 
#                   X, y, num_trials, num_folds, random_seed)
    
    feat_list = list()
    # feature set 1: by corr
    X1, feature_1 = extractFeatures(X,y,13)
    feat_list.append(feature_1)
    cvScore_1 = cross_validated_accuracy(T, 
                   X1, y, num_trials, num_folds, random_seed)
    print(f'feature 1 accuracy: {cvScore_1:.3f}')
    # feature set 2: by random
    X2, feature_2 = randomFeatures(X,13,random_seed)
    feat_list.append(feature_2)
    cvScore_2 = cross_validated_accuracy(T, 
                   X2, y, num_trials, num_folds, random_seed)
    print(f'feature 2 accuracy: {cvScore_2:.3f}')
    # feature set 3: by given example
    feature_3 = ['RIDAGEYR','BMXWAIST','BMXHT','LBXTC','BMXLEG','BMXWT','BMXBMI',
                 'RIDRETH1','BPQ020','ALQ120Q','DMDEDUC2','RIAGENDR','INDFMPIR']
    X3 = X[feature_3]
    feat_list.append(feature_3)
    cvScore_3 = cross_validated_accuracy(T, 
                   X3, y, num_trials, num_folds, random_seed)
    print(f'feature 3 accuracy: {cvScore_3:.3f}\n')
    
    # compare the accuracies and choose the best feature set
    best_index = np.argmax([cvScore_1,cvScore_2,cvScore_3])
    best_accuracy = max([cvScore_1,cvScore_2,cvScore_3])
    best_feat = feat_list[best_index]
    X_train = X[best_feat]
    best_ccp = automatic_dt_pruning(T, 
                    X_train, y, num_trials, num_folds, random_seed)
    
    print(f'\nbest feature set: feature_{best_index+1}')
    print(f'best accuracy: {round(best_accuracy,3)}')
    print(f'best feature:\n {best_feat.values}\n')
    print(f'best ccp_alpha: {round(best_ccp,2)}\n')
    
    # plot the tree
    dot_data = tree.export_graphviz(T, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("Unpruned Tree") 
    dot_data = tree.export_graphviz(T, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['None', 'Diabetic'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data) 
    graph
    
    # making predictions on unlabelled data set
    X_unlabel = pd.read_csv('hw1-NHANES-diabetes-test-unlabeled.csv')
    X_unlabel = preprocessing(X_unlabel,1)
    X_unlabel = X_unlabel[best_feat]
    T.ccp_alpha = best_ccp
    T = T.fit(X_train, y)
    y_pred = T.predict(X_unlabel)
    predDf = pd.Series(y_pred).rename('DIABETIC')
    predDf.to_csv('cis519_hw1_predictions.csv',header=False,index=False)
    print(f'Diabetic Ratio (Unabel): {y_pred.sum()/len(y_pred):.2f}\n')
    
    stop = timeit.default_timer()
    print(f'Time: {stop - start:.0f}s')