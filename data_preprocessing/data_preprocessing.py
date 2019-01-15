'''
Data Preprocessing recipes

Zac Hancock (zshancock@gmail.com)

'''

# Import Libraries needed. 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split # Note : formally cross_Validation, but that library is deprecated
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

'''
# Import dataset

data = pd.read_csv('../data.csv') # path to file of interest

# divide into x, y (features, labels)

x_features = data.iloc[:,:] # index appropriately
y_labels = data.iloc[:,:] # index appropriately

'''

# Dealing with NaN values in the data, if present

# Find out if the data contains NA values

data.isnull().any().any() # returns a boolean if any are present (Note: if true, consider addressing before the x, y division)
data.isnull().sum().sum() # returns the number of NaNs in the dataframe

# Removal of NaNs from dataframe (do before x_features, y_labels)

data.dropna() # drops incomplete rows
data.dropna(axis = 'columns') # drops any column with atleast 1 missing entry
df.dropna(subset=['feature1', 'feature2', ...]) # drops rows that have NAs in specific columns


# If NOT going to remove use Imputer

'''
Imputer
missing_values - 0 or NaN (default is NaN) 
strategy - how to replace NaN values (mean, median, most_frequent)
axis - 0 = columns (default), 1 = rows
'''

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x_features[:,:]) # index appropriately
x_features[:,:] = imputer.transform(x_features[:,:])


# Address any categorical features

# labelEncoder is good for binary (yes/no, etc) - or categories that are related (low/medium/high, etc.)

encoder_x = LabelEncoder()
X[:,:] = encoder_x.fit_transform(X[:,:]) #index appropriately (index to categorical features)

# if inappropriate to encode, OneHotEncode.

'''
OneHotEncode
categorical_features = which features are the categorical features? which index/column?

'''

one_hot = OneHotEncoder(categorical_features = [0]) # index appropriately (i.e. [0] is the first column)
x_features = one_hot.fit_transform(x_features).toarray()

# With an X, Y matrix (x - features, y - labels)

# how large is the hold-out data set, i.e. 20%

test_size = 0.2

# into X/Y Train/Test. 

X_train, X_test, y_train, y_test = train_test_split(x_features, y_labels, test_size = test_size, random_state = 0)

# Scaling features if necessary using StandScaler
# StandardScaler will return a distribution that has a mean = 0

scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test) # do not need to fit again (already fit on X_train)


# Now the labels for y_train (only necessary for regression problems, not Classification problems)
scale_y = StandardScaler()
y_train = scale_y.fit_transform(y_train)
y_test = scale_y.transform(y_test)


# Alternatively, scale using MinMaxScaler
# MinMaxScaler will return the Maximum value as 1 and the minimum values as 0, with all the other data falling between 0-1

x_minmax_scaler = MinMaxScaler()
X_train = x_minmax_scaler.fit_transform(X_train)
X_test = x_minmax_scaler.transform(X_test)

# If necessary (i.e. regression not classification.)
y_minmax_scaler = MinMaxScaler()
y_train = y_minmax_scaler.fit_transform(y_train)
y_test = y_minmax_scaler.transform(y_test)


