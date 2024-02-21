# custom_model.py

import tensorflow as tf
from keras import applications
from keras.engine import Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras_vggface.vggface import VGGFace

def get_custom_model():
    # Custom parameters
    learning_rate = 0.00001
    nb_class = 2
    hidden_dim = 512
    batch_size = 32
    img_height = 224
    img_width = 224

    # Load the pretrained ResNet50 model
    pretrained_model = applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3))

    # Model architecture
    x = Flatten()(pretrained_model.layers[-1].output)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim // 4, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    out = Dense(nb_class, activation='softmax', name='fc9')(x)

    # Create the custom model
    custom_pretrained_model = Model(pretrained_model.input, out)

    return custom_pretrained_model
