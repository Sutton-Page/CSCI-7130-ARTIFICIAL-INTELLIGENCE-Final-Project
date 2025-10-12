import tensorflow as tf
import keras
from model import create_mobilevit

train_ds = tf.keras.utils.image_dataset_from_directory(
    './memes/train',
    seed=123,
    image_size=(256,256),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    './memes/test',
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
    _, accuracy = mobilevit_xxs.evaluate(val_ds)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    return mobilevit_xxs



# Training the model
trained_model  = run_experiment(epochs)

tf.saved_model.save(trained_model, "mobilevit_xxs")

# Convert to TFLite. This form of quantization is called
# post-training dynamic-range quantization in TFLite.
converter = tf.lite.TFLiteConverter.from_saved_model("mobilevit_xxs")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops.
]
tflite_model = converter.convert()
open("mobilevit_xxs.tflite", "wb").write(tflite_model)



