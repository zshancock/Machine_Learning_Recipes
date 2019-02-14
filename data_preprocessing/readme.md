## Data Preprocessing Tools
### Zac Hancock (zshancock@gmail.com)

Note: Please note that these are meant to be referenced 'per block', I did not follow conventional 'PEP8' formatting with libraries/imports.

### Data Preprocessing Recipes included

* Various handling methods for NAs
* encoding categorical features
* splitting an X/Y matrix of data into Training and Testing data
* Both *standard scaling* (mean=0) and *minmaxscaling* (0-1 scale) of features

### Required libraries

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
```

### Additional Insight

Remember preprocessing is often very unique to each case, and its very important to identify whether its a regression problem or classification problem. 

I used a function from the **sklearn.model_selection** module called **train_test_split**, please note this used to be in the module **cross_validation** which is deprecated. 
