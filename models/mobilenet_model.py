from keras.api.applications import MobileNetV2
from keras.api.layers import GlobalAveragePooling2D, Dense, Lambda
from keras.api.models import Model
import tensorflow as tf

def load_mobilenet_model():
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)  # ✅ Dùng Lambda

    model = Model(inputs=base_model.input, outputs=x)
    return model
