from tensorflow.keras.utils import plot_model


def summarize_model(keras_model, model_name):
    keras_model.summary()
    plot_model(keras_model, to_file=f"model-structure/{model_name}.png", show_shapes=True)
