#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dqn_model_keras.py
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers,models
import numpy as np
'''
def build_dqn_model():
     Toplam input boyutu: 7036
    input_layer = keras.Input(shape=(7036,), name="state_input")

    x = layers.Dense(512, activation="relu")(input_layer)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(4, activation="softmax", name="weights")(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model
'''
def build_dqn_model(state_dim=7036, action_dim=84):
    inputs = layers.Input(shape=(state_dim,), name="state_input")
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(action_dim, activation="linear", name="q_values")(x)
    model = models.Model(inputs=inputs, outputs=output, name="dqn_model")
    return model


def encode_obs_dict(obs_dict):
    """
    PyTorch versiyonundaki gibi:
    - board: (15,15,30)
    - raf: (30,)
    - bonus: (15,15,1)
    - stok: (30,)
    - skor_farki: (1,)
    Bunları düzleştirip birleştirir.
    """
    board = obs_dict["board"].reshape(-1)   # 6750
    raf   = obs_dict["raf"].reshape(-1)     # 30
    bonus = obs_dict["bonus"].reshape(-1)   # 225
    stok  = obs_dict["stok"].reshape(-1)    # 30
    skor  = obs_dict["skor_farki"].reshape(-1)  # 1

    return np.concatenate([board, raf, bonus, stok, skor], axis=0)  # shape=(7036,)


# Test örneği
if __name__ == "__main__":
    model = build_dqn_model()
    model.summary()

    # Dummy tek bir gözlemle test et
    dummy_obs = {
        "board": np.zeros((15, 15, 30), dtype=np.float32),
        "raf": np.zeros((30,), dtype=np.float32),
        "bonus": np.zeros((15, 15, 1), dtype=np.float32),
        "stok": np.zeros((30,), dtype=np.float32),
        "skor_farki": np.zeros((1,), dtype=np.float32),
    }
    x = encode_obs_dict(dummy_obs).reshape(1, -1)
    weights = model.predict(x)
    print("Ağırlıklar:", weights)

