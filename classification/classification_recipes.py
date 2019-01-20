'''
Classification recipes
Zac Hancock(zshancock@gmail.com)

Recall that preprocessing needs to be conducted before performing these tasks. Generally the assumption is made that data is 
in (X_train, y_train)( X_test, y_test) format. Refer to regression_recipes.ipynb for regression code.

Contents: 
- Naive Bayes
- Decision Tree
- Random Forrest

'''

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Initialize the naive bayes classifier

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make a prediction

y_pred = nb_classifier.predict(X_test)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Decision Tree (note: Preprocessing does not need scaling, since decision tree algorithms do not use Euclidean Distance)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import graphviz

# Initialize the decision tree classifier

dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)

# Make a prediction

y_pred = dt_classifier.predict(X_test)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the decision tree

dot_data = tree.export_graphviz(dt_classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dt") 
dot_data = tree.export_graphviz(dt_classifier, out_file=None, filled = True, rounded= True)
graph = graphviz.Source(dot_data) 

# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Initialize the random forest classifier (n_estimators is the 'number of trees')

rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Make a prediction

y_pred = rf_classifier.predict(X_test)

# Confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)


