import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_dataset = train_data_gen.flow_from_directory("training_set", target_size=(64, 64), batch_size=32, class_mode="binary")

test_datagen = ImageDataGenerator(rescale= 1./255)
test_dataset = test_datagen.flow_from_directory("test_set", target_size=(64, 64), batch_size=32, class_mode="binary")

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

cnn.fit(x = train_dataset, validation_data = test_dataset, epochs=10)

import numpy as np
import keras.utils as image
test_image = image.load_img('test_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict (test_image)

train_dataset.class_indices

if result[0][0] == 1:
    prediction = "White"
else:
    prediction = "Crimson"
    
print(prediction)