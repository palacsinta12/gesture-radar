import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input_layer")

    # Block 1
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # Block 2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # FLATTEN preserves the (Time x Bins) coordinate system for directional gestures
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="FMCW_Gesture_CNN")
    return model