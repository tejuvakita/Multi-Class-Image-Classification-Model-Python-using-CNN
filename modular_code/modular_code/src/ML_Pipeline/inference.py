import tensorflow as tf
import numpy as np
from .admin import model_path
from tensorflow.keras.preprocessing import image

# create an inference class 
class Inference:

    def __init__(self):
        self.img_size = 224
        self.model = self.load_model()
        self.label = ['driving_license', 'social_security', "others"]

    def load_model(self):
        """loads model from load_model from a given model path"""
        model = tf.keras.models.load_model(model_path)
        return model

    def infer(self, filename):
        """
        loads image ,resize it to 224*224 ,expan the dimension to have 4D and predicts using model loaded
        """
        img1 = image.load_img(filename, target_size=(self.img_size, self.img_size))
        Y = image.img_to_array(img1)
        X = np.expand_dims(Y, axis=0)
        val = np.argmax(self.model.predict(X))
        class_predicted = self.label[int(val)]
        return class_predicted
