## Classification Recipes
### Zac Hancock (zshancock@gmail.com)

Note: Please note that these are meant to be referenced 'per block' meaning, I did not follow conventional 'PEP8' formatting with 
libraries/imports. 

### Classification Tasks

* Naive Bayes (Gaussian, Multinomial)
* Decision Trees
* Random Forest

* Confusion Matrix included for quick evaluataions. 

### Required libraries

**Naive Bayes**
```
from sklearn.naive_bayes import GaussianNB, MultinomialNB
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
and a **.pdf** render of the decision tree. Example shown below.

![alt text](https://github.com/zshancock/Machine_Learning_Recipes/blob/master/classification/graphics/decisiontreegraphic.JPG)
