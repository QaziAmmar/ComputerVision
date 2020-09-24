import tensorflow as tf


inp = tf.keras.layers.Input(shape=(9,))

hidden = tf.keras.layers.Dense(5, activation="relu")(inp)
out = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss=tf.losses.mean_squared_error, metrics=['accuracy'])
model.summary()

