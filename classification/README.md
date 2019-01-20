## Classification Recipes
### Zac Hancock (zshancock@gmail.com)

### Classification Tasks

* Naive Bayes
* Decision Trees
* Random Forest

### Required libraries

**Naive Bayes**
```
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
```

**Decision Tree**
```
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import graphviz
```

**Random Forest**
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
```


### Additional Insight

Visualizing decision trees is notoriously difficult with graphviz in Python. My recipe provides an **inline** plot with Juptyer notebook 
and a **.pdf** render of the decision tree. 
