from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input

def create_diabetes_model(num_feature):
    input = Input(shape=(num_feature), name='input-layer')
    x = Dense(16, activation='relu', name='hidden-1')(input)
    x = Dense(8, activation='relu', name='hidden-2')(x)
    output = Dense(1, activation='sigmoid', name='output-layer')(x)

    model = Model(inputs=input, outputs=output)
    return model
