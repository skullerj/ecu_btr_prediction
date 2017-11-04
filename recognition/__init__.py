import keras
import numpy as np
from keras.models import load_model
from keras import applications

model = load_model('recognition/final_model.h5')

def predict(data):
    data = np.resize(data,(1,100,200,3))
    data = data*(1/255)
    result = model.predict(data);
    return np.amax(result),result[0].tolist().index(np.amax(result))
