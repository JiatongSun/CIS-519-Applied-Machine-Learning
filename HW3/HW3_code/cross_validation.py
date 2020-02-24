import numpy as np
from sklearn.model_selection import KFold

def cross_validated_accuracy(model, X, y, num_trials, num_folds):
    accuracy_list = []
    for trial in range(num_trials):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=trial)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
    
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            y_pred = y_pred[0].astype('O')
            accuracy = (y_test.values == y_pred.values).sum()/len(y_test)
            accuracy_list.append(accuracy)
            print(accuracy)
    cvScore = np.mean(accuracy_list)
    return cvScore