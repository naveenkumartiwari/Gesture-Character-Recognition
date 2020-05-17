
from __future__ import print_function
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from mnist import MNIST
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import model_from_json



img_rows, img_cols = 28, 28

emnist_data = MNIST(path='data1\\', return_type='numpy')
emnist_data.select_emnist('letters')
X, y = emnist_data.load_training()


X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)

y = y-1

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255
batch_size = 128
num_classes = 26
epochs = 10


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])




model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# Save the model weights for future reference
model_json = model.to_json()
with open("training_model.json", "w") as json_file:
    json_file.write(model_json)
model.save('training_model.h5')


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])