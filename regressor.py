import pickle
import numpy as np


def predict(model, X):
    prediction = model.predict(X)
    return np.exp(prediction)


def load_model(model_path):
    with open(model_path, 'rb') as src:
        model = pickle.load(src)

    return model
