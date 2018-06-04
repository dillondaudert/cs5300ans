# load and import cifar10 data
import pickle
import numpy as np
import tensorflow as tf

def app():
    # train and evaluate a dense neural network on CIFAR-10 using tf.keras

    # x samples are [samples, width, height, channels]
    # y's are [class_index]
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # create a random validation set
    perm = np.random.permutation(np.arange(x.shape[0]))
    num_train = int(x.shape[0] * .8)

    x_train = x[0:num_train,:]
    y_train = y[0:num_train]

    x_valid = x[num_train:,:]
    y_valid = y[num_train:]

    num_pixels = x.shape[1]*x.shape[2]*x.shape[3]

    # reshape to flatten out the images in this case
    x_train = x_train.reshape(-1, num_pixels)
    x_valid = x_valid.reshape(-1, num_pixels)
    x_test = x_test.reshape(-1, num_pixels)

    # convert the class integers to one-hots
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(num_pixels,)))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="rmsprop")

    # use the model
    model.fit(x_train, y_train, epochs=2, validation_data=(x_valid, y_valid), shuffle=True)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(model.predict(x_test[0:1,:]))


if __name__ == "__main__":
    app()
