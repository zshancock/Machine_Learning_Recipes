'''
Data preprocessing - Working with Text

Zac Hancock (zshancock@gmail.com)

'''
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# several commonly used vectorizer setting - stop words set to english for all. 

#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

#  unigram term frequency vectorizer, set minimum document frequency to 5
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english')

#  unigram and bigram term frequency vectorizer, set minimum document frequency to 5
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=5, stop_words='english')

#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words='english')

# With previously vectorization selection - transform into a dataframe

# for this example - I used the default CountVectorizer()

input_corpus = 'This is where the data is read, this should be in list format, CountVectorizer has a built in tokenizer.'
vectorizer = CountVectorizer(stop_words = 'english') # replace with above variants if desired.
 

# fit transform does both steps in one, alternatively each can be called individually (.fit(), .transform())   
vectorizer_results = vectorizer.fit_transform(input_corpus)
columns = vectorizer_results.get_feature_names()
vectorized_df = pd.DataFrame(vectorizer_results.toarray(), columns = columns)

# Finalize preparation by splitting into X_train, etc...

# Labels in the split would be if this data was examining Reviews and the label was sentiment, or answers on 
# a test and the labels were right/wrong, etc.

# split the data into 80 - 20%, random seed set to 0. (can be any number, just be consistent)

X_train, X_test, y_train, y_test = train_test_split(vectorized_df, labels, test_size=0.2, random_state=0)

# Examine shape. 
print('Shape of Training ' ,  X_train.shape)
print('Shape of Testing ' , X_test.shape)

# Data is ready for analysis by any number of sklearn algorithms (MultinomialNB, BernoulliNB, etc.) 
# Be sure that your vectorizer is appropriate (i.e. BernoulliNB expects binary = True, whereas Multinomial expects frequencies)

