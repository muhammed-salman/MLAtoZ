# Convolutional Neural Network

# Install Theano, Tensorflow & Keras

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#part 1 Building the CNN

#Init the CNN
classifier = Sequential()

#step-1 : convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation ="relu"))

#step-2 : Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Second convolution layer to improve accuracy (Deeper Deep Learning)
classifier.add(Convolution2D(32,(3,3),activation ="relu")) #when you add a new covolution layer no need for input_shape parameter

classifier.add(MaxPooling2D(pool_size = (2,2)))

#step-3 : Falttening
classifier.add(Flatten())

#step-4 : Full Connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#step-5 : Compiling CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#part 2 Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)