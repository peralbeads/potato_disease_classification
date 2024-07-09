import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import dataset
import numpy as np
from dataset import Dataset

resize_and_rescale = dataset.resize_and_rescale
data_augmentation = dataset.data_augmentation
val_ds = dataset.val_ds
train_ds = dataset.train_ds

BATCH_SIZE = Dataset.BATCH_SIZE
IMG_SIZE = Dataset.IMG_SIZE
CHANNELS = Dataset.CHANNELS
Epochs = Dataset.EPOCHS


class Model:
    def __init__(self):
        pass

    data = Dataset
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    n_classes = 3
    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])

    model.build(input_shape=(BATCH_SIZE,) + input_shape)
    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=Epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )

    scores = model.evaluate(dataset.test_ds)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(Epochs), acc, label='Accuracy')
    plt.plot(range(Epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(Epochs), loss, label='loss')
    plt.plot(range(Epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    for images_batch, labels_batch in dataset.test_ds.take(1):

        first_image = images_batch[0].numpy().astype('uint8')
        first_label = labels_batch[0].numpy()

        print("first image to predict")
        plt.imshow(first_image)
        print("actual label: ", Dataset.class_names[first_label])

        batch_prediction = model.predict(images_batch)
        print("predicted label: ", Dataset.class_names[np.argmax(batch_prediction[0])])

    @staticmethod
    def predict(model, img):
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # create a batch
        predictions = model.predict(img_array)

        predicted_class = Dataset.class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence

    plt.figure(figsize=(15, 15))
    for images, labels in dataset.test_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class, confidence = predict(model, images[i].numpy())
            actual_class = Dataset.class_names[labels[i]]
            plt.title(f"Actual: {actual_class}, \n Predicted: {predicted_class}, \n Confidence: {confidence}%")
            plt.axis("off")

    plt.show()
