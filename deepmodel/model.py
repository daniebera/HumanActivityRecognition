import json as js
import numpy as np
import cv2

from tensorflow.python.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.layers import Dense, Activation, Dropout, Bidirectional
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.optimizers import Adam

import config.config as cfg



class DeepModel:
    def __init__(self):
        # Loading label-id mapping and reverse mapping
        with open(cfg.LABEL_ID_PATH) as json_file:
            self.labels = js.load(json_file)
        with open(cfg.ID_LABEL_PATH) as json_file:
            self.labels_idx2word = js.load(json_file)
        self.expected_frames = 7
        self.nb_classes = len(self.labels)
        self.model_extractor = self.feature_extractor(include_top=False)
        self.model_classifier = self.feature_classifier(cfg.WEIGHT_PATH)

    # Creation of VGG16 model with ImageNet weights

    def feature_extractor(self, include_top=False):
        vgg16_model = VGG16(include_top=include_top, weights='imagenet')
        vgg16_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return vgg16_model

    # Creation of Bi-LSTM model with weights

    def feature_classifier(self, weight_file_path):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=cfg.HIDDEN_UNITS, return_sequences=True),
                                input_shape=(cfg.EXPECTED_FRAMES, cfg.NUM_INPUT_TOKENS)))
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.load_weights(weight_file_path)

        return model

    def get_prediction(self, data):
        x_samples = self.extract_vgg16_features_live(self.model_extractor, data)
        # Resize feature length
        frames = x_samples.shape[0]
        # Cut off the exceeding frames w.r.t. the expected frames number (Why to cut off ?)
        if frames > cfg.EXPECTED_FRAMES:
            x_samples = x_samples[0:cfg.EXPECTED_FRAMES, :]
        # Append frames with zero-features to reach the expected frames number (Why zero-features ?)
        elif frames < cfg.EXPECTED_FRAMES:
            temp = np.zeros(shape=(cfg.EXPECTED_FRAMES, x_samples.shape[1]))
            temp[0:frames, :] = x_samples
            x_samples = temp
        # Get for each prediction the index of the predicted class
        y_pred = (np.argmax(self.model_classifier.predict(np.array([x_samples]))[0]))
        y_pred = self.labels_idx2word[str(y_pred)]
        return y_pred

    # Extract single-video features from VGG16 on the fly

    def extract_vgg16_features_live(self, model, video_input_file_path):
        print('Extracting frames from video: ', video_input_file_path)
        vidcap = cv2.VideoCapture(video_input_file_path)
        success, image = vidcap.read()
        features = []
        success = True
        count = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = model.predict(input).ravel()
                features.append(feature)
                count = count + 1
        x_samples = np.array(features)
        return x_samples
