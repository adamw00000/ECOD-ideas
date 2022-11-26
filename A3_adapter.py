import os
import atexit
import tempfile
import numpy as np
import tensorflow as tf

from A3.libs.A3 import A3
from A3.libs.architecture import dense_ae, alarm_net, VariationalAutoEncoder, RandomNoise

class A3Adapter():
    def __init__(self, max_target_epochs=30, max_a3_epochs=60, patience=30, verbose=0, model_dir=None):
        self.target_epochs = max_target_epochs
        self.a3_epochs = max_a3_epochs
        self.verbose = verbose
        self.patience = patience
        self.model_dir = model_dir

    def fit(self, X_train):
        N = len(X_train)
        val_indices = np.random.choice(range(N), size=int(0.1 * N), replace=False)
        val_mask = np.where(np.isin(range(N), val_indices), True, False)
        X_train, X_val = X_train[~val_mask], X_train[val_mask]

        train_target = (X_train, X_train)
        val_target = (X_val, X_val) # fake!

        train_alarm = (X_train, np.zeros(len(X_train)))
        val_alarm = (X_val, np.zeros(len(X_val))) # fake!

        TRAIN_TARGET = True

        random_noise = RandomNoise("normal")
        model_vae = VariationalAutoEncoder(
            input_shape=X_train.shape[1:],
            layer_dims=[800, 400, 100, 25]
        )
        model_vae.compile(optimizer=tf.keras.optimizers.Adam(.001))
        # Subclassed Keras models don't know about the shapes in advance... build() didn't do the trick
        model_vae.fit(train_target[0], epochs=0, batch_size=256)

        if TRAIN_TARGET:
            with CustomNamedTemporaryFile(prefix='A3_target_', suffix='.hdf5', delete=True) as tmp:
                MODEL_TARGET_PATH = tmp.name
                # MODEL_TARGET_PATH = os.path.join(self.model_dir, 'best_model.hdf5')

                mc_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_TARGET_PATH, monitor='val_loss',mode='min', save_best_only=True)
                es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)

                # Create target network
                model_target = dense_ae(input_shape=X_train.shape[1:], layer_dims=[1000, 500, 200, 75])
                # model_target.compile(optimizer='adam', loss='binary_crossentropy')
                model_target.compile(optimizer='adam', loss='mse')
                history_target = model_target.fit(
                    train_target[0], train_target[1],
                    validation_data=val_target,
                    epochs=self.target_epochs, batch_size=256, 
                    callbacks=[
                        es_callback, 
                        # mc_callback,
                    ],
                    verbose=self.verbose,
                )
                epochs_target = len(history_target.history['loss'])

                # model_target.load_weights(MODEL_TARGET_PATH)
                # os.remove(MODEL_TARGET_PATH)

            # Create alarm and overall network
            model_a3 = A3(target_network=model_target)
            model_a3.add_anomaly_network(random_noise)
            model_alarm = alarm_net(
                layer_dims=[1000, 500, 200, 75],
                input_shape=model_a3.get_alarm_shape(),
            )
            model_a3.add_alarm_network(model_alarm)

        with CustomNamedTemporaryFile(prefix='A3_a3_', suffix='.hdf5', delete=True) as tmp:
            MODEL_A3_PATH = tmp.name
            # MODEL_A3_PATH = os.path.join(self.model_dir, 'best_model.hdf5')
            mc_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_A3_PATH, monitor='val_loss',mode='min', save_best_only=True)
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)

            model_a3.compile(
                optimizer=tf.keras.optimizers.Adam(.00001),
                loss="binary_crossentropy",
            )
            history_a3 = model_a3.fit(
                train_alarm[0],
                train_alarm[1],
                validation_data=val_alarm,
                epochs=self.a3_epochs, batch_size=256, 
                callbacks=[
                    es_callback, 
                    # mc_callback,
                ],
                verbose=self.verbose,
            )
            epochs_a3 = len(history_a3.history['loss'])

            # model_a3.load_weights(MODEL_A3_PATH)
            # os.remove(MODEL_A3_PATH)

        self.model = model_a3
    
    def score_samples(self, samples):
        return 1 - self.model.predict(samples).reshape(-1)


class CustomNamedTemporaryFile:
    """
    This custom implementation is needed because of the following limitation of tempfile.NamedTemporaryFile:

    > Whether the name can be used to open the file a second time, while the named temporary file is still open,
    > varies across platforms (it can be so used on Unix; it cannot on Windows NT or later).
    """
    def __init__(self, mode='wb', prefix='', suffix='', delete=True):
        self._mode = mode
        self._delete = delete
        self._prefix = prefix
        self._suffix = suffix

    def __enter__(self):
        # Generate a random temporary file name
        file_name = os.path.join(tempfile.gettempdir(), 
            f'{self._prefix}{os.urandom(24).hex()}{self._suffix}')
        # Ensure the file is created
        open(file_name, "x").close()
        # Open the file in the given mode
        self._tempFile = open(file_name, self._mode)

        self.name = file_name
        atexit.register(self.__exit__)

        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempFile.close()
        if self._delete:
            os.remove(self._tempFile.name)
        atexit.unregister(self.__exit__)