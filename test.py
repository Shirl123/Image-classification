

#predicting single images
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single_prediction\1-with-mask.jpeg',target_size=(64,64))
test_image = image.image_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction='mask'
else:
    prediction='No mask'
print(prediction)
