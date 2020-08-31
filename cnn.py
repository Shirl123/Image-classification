#data preprocessing
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale =1./255,
                                    shear_range =0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
training_set = train_datagen.flow_from_directory('train',target_size = (64,64),batch_size=32,class_mode ='binary')


test_datagen = ImageDataGenerator(rescale=1./255)
testing_set = test_datagen.flow_from_directory('test',target_size=(64,64),batch_size=32,class_mode='binary')

#creating a cnn
cnn = tf.keras.models.Sequential()
# Adding first convolution layer
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3,activation='relu',input_shape=[64,64,3]))
#Adding first maxpool layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides=2))

#Adding second set of convolution and maxpool layers
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Adding flattening layer
cnn.add(tf.keras.layers.Flatten())
#Adding fullyconnected layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#Adding output layer which classifies
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#Training the CNN on training set
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Training the CNN on the training set and evaluating the CNN on the test set
cnn.fit(x = training_set,validation_data = testing_set,epochs=25)


test_image = image.load_img('1-with-mask.jpeg',target_size=(64,64))
test_image = image.image_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction='mask'
else:
    prediction='No mask'
print(prediction)
