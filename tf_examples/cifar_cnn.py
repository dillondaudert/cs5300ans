# load and import cifar10 data
import pickle
import numpy as np
import tensorflow as tf

def app():
    # train and evaluate a CNN on CIFAR-10 using tf.keras

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

    # convert the class integers to one-hots
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, 4, activation=tf.nn.relu, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, 4, activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # flatten
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    optim = tf.keras.optimizers.SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optim)

    # use the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid), shuffle=True)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(model.predict(x_test[0:1,:]))


if __name__ == "__main__":
    app()
