
## This program builds a CNN model to predict synthesised digits (0-9)
## which are superimposed on various and unuque backgrounds. Model is trained
## with 10k images.

# set working directory
import os
os.chdir('/Users/Bradley/Dropbox/...')

# import the Keras library
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

##########
## Build the model
##########

# CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

# subsequent ANN
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 10, activation = 'softmax'))

# compile
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

##########
## Read in and prepare the data
##########

# training data
train_prep = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)
training_set = train_prep.flow_from_directory('synthetic_digits/training_set',
                                              target_size = (64, 64),
                                              batch_size = 32)

# test data
test_prep = ImageDataGenerator(rescale = 1./255)
test_set = test_prep.flow_from_directory('synthetic_digits/test_set',
                                         target_size = (64, 64),
                                         batch_size = 32)

##########
## Run the model
##########

model.fit_generator(training_set,
                    steps_per_epoch = 10000,
                    epochs = 3,
                    validation_data = test_set,
                    validation_steps = 2000)
