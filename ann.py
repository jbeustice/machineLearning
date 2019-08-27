
## This program builds and tunes an ANN model to predict whether a warehouse
## customer is in the retail industry or the restaurant business. About 500
## observations. 

# set working directory
import os
os.chdir('/Users/Bradley/Dropbox/...')

##########
## Read in, split, and standardize the data
##########

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
dataset = pd.read_csv('wholesale.csv')
X = dataset.iloc[:, 2:len(dataset.columns)]
y = dataset.iloc[:, 0]

# encode categorical variables
y = pd.get_dummies(y,columns=['Channel'],prefix='')
y = y.drop('_hotrest',axis=1)

# split data --> training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##########
## Build and tune model
##########

# import libraries for ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# build ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 4,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         input_dim = 6))
    classifier.add(Dense(units = 1,
                         kernel_initializer = 'uniform',
                         activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer,
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

# tune ANN
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [20, 32],
              'epochs': [100, 250],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)

# run ANN
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_acc = grid_search.best_score_

##########
## Prepare and test best model
##########

# build best ANN
best_mod = Sequential()
best_mod.add(Dense(units = 4,
                   kernel_initializer = 'uniform',
                   activation = 'relu',
                   input_dim = 6))
best_mod.add(Dense(units = 1,
                   kernel_initializer = 'uniform',
                   activation = 'sigmoid'))
best_mod.compile(optimizer = best_params['optimizer'],
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])

# run best ANN
best_mod.fit(X_train,
             y_train,
             batch_size = best_params['batch_size'],
             epochs = best_params['epochs'])

# predict on test set
from sklearn.metrics import confusion_matrix
y_pred = best_mod.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
final_acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
final_acc