# In model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def get_model(input_shape):
    # Define the input layer with the shape of the data
    input_layer = Input(shape=input_shape)

    # Convolutional layers
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flatten = Flatten()(pool2)

    # Age prediction layer
    age_layer = Dense(128, activation='relu')(flatten)
    age_output = Dense(1, activation='sigmoid', name='age_output')(age_layer)

    # Gender prediction layer
    gender_layer = Dense(128, activation='relu')(flatten)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=[age_output, gender_output])

    # Compile model
    model.compile(optimizer='adam',
                  loss={'age_output': 'binary_crossentropy', 'gender_output': 'binary_crossentropy'},
                  metrics={'age_output': 'accuracy', 'gender_output': 'accuracy'})

    return model
