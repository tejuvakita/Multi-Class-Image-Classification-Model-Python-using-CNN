from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from .admin import model_path
import tensorflow as tf
import cv2
import os
import numpy as np

#create a class for classifier model
class Classifier:
    def __init__(self, train_dir):
        """
        :param train_dir:train dir with folders specified below in self.label few of the variables like imz_size,
        epochs,datagen object,tensorboard object for callback etc has been initialized in the __init__
        """
        self.label = ['driving_license', 'social_security', "others"]
        self.img_size = 224
        self.epochs = 10
        self.train_dir = train_dir
        self.datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        log_dir = "./logs"
        # defines callback for tensorboard to be added in model.fit ,there are few parameters
        # 1) log_dir: The folder where all the events files will be saved for tensorboard
        # 2) histogram_freq: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model.
        # 3) write_images	whether to write model weights to visualize as image in TensorBoard.
        # 4) write_steps_per_second	whether to log the training steps per second into Tensorboard. This supports both epoch and batch frequency logging.
        # 5) update_freq	'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000, the callback will write the metrics and losses to TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow down your training.
        # 6) profile_batch	Profile the batch(es) to sample compute characteristics. profile_batch must be a non-negative integer or a tuple of integers. A pair of positive integers signify a range of batches to profile. By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
        # 7) embeddings_freq	frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized.
        # 8) embeddings_metadata	Dictionary which maps embedding layer names to the filename of a file in which to save metadata for the embedding layer. In case the same metadata file is to be used for all embedding layers, a single filename can be passed.
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                                   write_images=False,
                                                                   update_freq='epoch',
                                                                   profile_batch=2, embeddings_freq=1,
                                                                   embeddings_metadata=None)

    def train(self):
        """
        calls sel.model to get an model instance,initializes optimizers,,compile it and trains on the data got from get_data method.
        """
        model = self.model()
        opt = Adam(learning_rate=0.001) # initialise the optimizer
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy']) #compile the model

        train_data = self.get_data(self.train_dir)

        x_train = [] #create list for x_train data
        y_train = [] #create list for _train data

        for feature, label in train_data:
            x_train.append(feature) #appending the lists
            y_train.append(label)

        # Normalize the data
        x_train = np.array(x_train) / 255
        x_train.reshape(-1, self.img_size, self.img_size, 1)
        y_train = np.array(y_train)

        self.datagen.fit(x_train)
        history = model.fit(x_train, y_train, epochs=self.epochs, callbacks=[self.tensorboard_callback])
        print(history)
        model.save(model_path)

    def get_data(self, data_dir):
        """
        :param data_dir: training dir to read image,resize it to 224*224 as specified
        :return: return as a numpy array of image and class
        """
        data = []
        for each_label in self.label:
            path = os.path.join(data_dir, each_label)
            class_num = self.label.index(each_label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)

    def model(self):
        """
        adds a bunch of sequential,conv and maxpool layers,before adding a dense layer of 3 for 3 labels after flattening it out.
        :return: model object
        """
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(3, activation="softmax"))

        return model
