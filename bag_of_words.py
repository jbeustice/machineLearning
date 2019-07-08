
## This program develops multiple bag-of-words models to classify IMDb movie
## reviews as positive or negative. A sample set of 50k reviews are used to
## train and validate the models (NB and SVM) using both word occurrence and
## TF-IDF algorithms. 

# import library
import os

# set working directory
os.chdir('/Users/Bradley/Desktop/aclImdb')
cwd = os.getcwd()


##########
## Read in the data and split into training/test
##########

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# function --> reads in individual text files in a folder and attaches sentiment
def f_ReadIn(folder,sentiment):
    result = []
    os.chdir(os.path.join(cwd,folder))
    f_cwd = os.getcwd()
    for f in os.listdir(f_cwd):
        if os.path.isfile(f) and f.endswith(".txt"):
           with open(os.path.join(f_cwd,f),'r') as file:
                content = file.read()
           result.append([content,sentiment])
    return pd.DataFrame(result)

# call function f_ReadIn() and combines into one df
neg = f_ReadIn('negative',0)
pos = f_ReadIn('positive',1)
df = pd.concat([neg,pos],ignore_index=True)
df.columns = ['review','sentiment']
df.head()

# free up memory
del neg, pos

# reset working directory
os.chdir(cwd)

# split data into train and test sets --> 70/30
X_train, X_test, y_train, y_test = train_test_split(df['review'],df['sentiment'],test_size = 0.3)


##########
## Create vector of features using the count and TF-IDF algorithms
##########

# import libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# count occurrence
c_vectorizer = CountVectorizer(stop_words='english',max_features=1000)
c_train = c_vectorizer.fit_transform(X_train)
c_test= c_vectorizer.transform(X_test)

# TF-IDF
tf_vectorizer = TfidfVectorizer(stop_words='english',max_features=1000)
tf_train = tf_vectorizer.fit_transform(X_train)
tf_test = tf_vectorizer.transform(X_test)


##########
## Build and tune models (Naive Bayes and SVM)
##########

# import libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# function --> tune NB
def f_tune_nb(X,y):
    alphas = [0.1,0.5,1]
    param_grid = {'alpha':alphas}
    grid_search = GridSearchCV(MultinomialNB(),param_grid,cv=5)
    grid_search.fit(X,y)
    return grid_search.best_params_

# function --> tune SVM (not feasible on a local computer)
def f_tune_svm(X,y):
    costs = [0.1,1,10]
    gammas = [0.1,1,4]
    param_grid = {'C':costs,'gamma':gammas}
    grid_search = GridSearchCV(svm.SVC(),param_grid,cv=5)
    grid_search.fit(X,y)
    return grid_search.best_params_

# print tuned parameters
print('')
print('c_nb --> %a' % f_tune_nb(c_train,y_train))
print('tf_nb --> %a' % f_tune_nb(tf_train,y_train))
print('c_svm --> %a' % f_tune_svm(c_train,y_train))
print('tf_svm --> %a' % f_tune_svm(tf_train,y_train))


##########
## Run tuned models (Naive Bayes and SVM) and compare results
##########

# NB count with tuned parameter
c_nb = MultinomialNB(alpha=0.1).fit(c_train,y_train)
c_nb_pred = c_nb.predict(c_test)

# NB TF-IDF with tuned parameter
tf_nb = MultinomialNB(alpha=0.5).fit(tf_train,y_train)
tf_nb_pred = tf_nb.predict(tf_test)

# SVM count with tuned parameters
c_svm = svm.SVC(C=1,gamma=0.1)
c_svm.fit(c_train,y_train)  
c_svm_pred = c_svm.predict(c_test)

# SVM TF-IDF with tuned parameters
tf_svm = svm.SVC(C=1,gamma=0.1)
tf_svm.fit(tf_train,y_train)  
tf_svm_pred = tf_svm.predict(tf_test)

# print model accuracy
print('')
print('Using Count Vectorizer and Naive Bayes')
print('Accuracy: %0.4f' % (accuracy_score(y_test,c_nb_pred)))
print('')
print('Using TF-IDF Vectorizer and Naive Bayes')
print('Accuracy: %0.4f' % (accuracy_score(y_test,tf_nb_pred)))
print('')
print('Using Count Vectorizer and SVM')
print('Accuracy: %0.4f' % (accuracy_score(y_test,c_svm_pred)))
print('')
print('Using TF-IDF Vectorizer and SVM')
print('Accuracy: %0.4f' % (accuracy_score(y_test,tf_svm_pred)))
