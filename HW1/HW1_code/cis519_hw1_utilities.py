import random 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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