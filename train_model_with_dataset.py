import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import keras
from model import create_mobilevit
import matplotlib.pyplot as plt



train_ds = tf.keras.utils.image_dataset_from_directory(
  './memes',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(256,256),
  batch_size=32)


val_ds = tf.keras.utils.image_dataset_from_directory(

    './memes',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256,256),
    batch_size=32)

train_ds = train_ds.map(

    lambda image, label: (image, tf.one_hot(label,depth=2)
                          )
    )

val_ds = val_ds.map(

    lambda image, label: (image,tf.one_hot(label,depth=2)
                          )
    )

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

learning_rate = 0.002
label_smoothing_factor = 0.1
epochs = 30
num_classes= 2

batch_size = 64
auto = tf.data.AUTOTUNE

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_factor)



def run_experiment(epochs=epochs):
    mobilevit_xxs = create_mobilevit(num_classes=num_classes)
    mobilevit_xxs.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5`
    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    mobilevit_xxs.fit(train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_callback],
    )
    mobilevit_xxs.load_weights(checkpoint_filepath)
    _, accuracy = mobilevit_xxs.evaluate(val_dataset)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    return mobilevit_xxs



