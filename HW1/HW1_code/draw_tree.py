import graphviz
import pandas as pd
from sklearn import tree

data = {'PainLocation':['head','head','head','head','head','extremeties',
                        'extremeties','extremeties','extremeties','extremeties',
                        'chest','chest','chest','chest'], 
        'Temperature':[98,97,98,98,97,98,102,101,98,98,97,104,98,100],
        'Sp02':[96,98,95,92,80,80,98,95,97,90,70,98,98,90],
        'Tachycardic':['false','false','true','true','true','true','true',
                       'false','false','false','true','false','true','false'],
        'Class':['OutPatient','OutPatient','Admit','Admit','Admit','Admit',
                 'OutPatient','OutPatient','OutPatient','Admit','Admit',
                 'Admit','Admit','Admit']}
df = pd.DataFrame(data)
X_train = df.drop('Class', axis = 1)
y_train = df['Class']
X_train = pd.get_dummies(X_train)

# Train the decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)

# Plot the tree
tree.plot_tree(clf)

print()
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Unpruned Tree") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X_train.columns,  
                     class_names=['Admit', 'OutPatient'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 
graph                                                 # display the graph

