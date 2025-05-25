import tensorflow as tf

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model
