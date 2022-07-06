#import the necessary packages
import nsemble.config as config
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def build_shallow_net():
    # define the model
    model = Sequential()
    model.add(Dense(config.denseUnits, input_dim=10, activation="relu"))
    model.add(Dense(config.denseUnits, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # compile the keras model
    model.compile(loss="binary_crossentropy", optimizer="adam",
        metrics=["accuracy"]
    )
    # return model
    return model