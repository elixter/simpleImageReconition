from PIL import Image

#cat_image_pathname = 'C:/Users/ltw97/Desktop/hoya.jpg'
#cat_image = Image.open(cat_image_pathname)
#cat_image.show();

#display_image_pathname = input('Enter image pathname: ')
#display_image = Image.open(display_image_pathname)
#display_image.show()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', ' ship', 'truck']

from keras.datasets import cifar10

## tuples have image arrays and labels first is for training, second is for testing
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#index = int(input('enter an image index: '))
#display_image = x_train[index] # it has pixel values
#display_label = y_train[index][0] # [0] is for attirbuting number itselves

#from matplotlib import pyplot as plt


#red_image = image.fromarray(display_image)
#red, green, blue = red_image.split()

#plt.imshow(red, cmap="Reds")
#plt.show()

from keras.utils import np_utils

new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')

# make number 0 to 1
new_X_train /= 255
new_X_test /= 255
new_Y_train = np_utils.to_categorical(y_train)
nwe_Y_test = np_utils.to_categorical(y_test)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

model = Sequential()

# convolution2D model to help to ensure that the orientation of 
# the image isn't going to effect anything
# it's not going to have a negative effect on the accurcy of our recognizer
# as well to specify several crucial peice of information
# such as the input shaepe the activation function we're going to be using.
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))

# To make sure that our image isn't going to ignore
# any parts of our image so 2 X 2 size and shows that it's going to
# cover pretty much every part of the image to give a slightly
# more accurate prediction
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout which want only one the arrays as we kind of dealing with 
# data retriveing from is going to be basically a 32 X 32 matrix
model.add(Flatten())

# dense layers of the head to basically try to remember
# which characteristics make up each kind of category of images
# so, distinguish featchers
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))   # 32 * 32 = 512

# ensure taht not only are we going to too much detail and ignoring the big picture
# but that the model doesn't become lazy enough to try to forget new examples
# kind of help it to pay attention to every new example every new training
# piece of data that comes in
model.add(Dropout(0.5))

# just for an extra degree of accuracy kind of approach it using
# a couple of different activation functions
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

# For save our model
import h5py
model.save('Trained_model.h5')
