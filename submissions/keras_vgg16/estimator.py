import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, to_categorical

from joblib import Parallel, delayed

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


num_classes = 403

def _load_image(filename, transforms):
    image = tf.image.decode_jpeg(tf.io.read_file(filename))
    for tr in transforms:
        image = tr(image)
    return image.numpy()


def train_valid_split(X, y, valid_size=0.3, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    n_samples_valid = int(valid_size * len(X))
    idx_train, idx_valid = indices[:-n_samples_valid], indices[-n_samples_valid:]
    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    return X_train, X_valid, y_train, y_valid


def transform_image(x):
    if x.shape[2] == 4:
        x = x[:, :, :3]
    x = tf.image.resize_with_crop_or_pad(x, 1024, 1024)
    x = tf.image.resize(x, (224, 224))
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x


class DataGenerator(Sequence):

    def __init__(self, X_paths, y=None, batch_size=1, n_channels=3,
                 num_classes=num_classes, shuffle=True,
                 transforms=(), preload=False, n_jobs=10):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.transforms = transforms
        self.n_channels = n_channels
        self.X_paths = X_paths
        self.preload = preload
        self._dim = _load_image(self.X_paths[0], self.transforms).shape[:2]

        if preload:
            print('Preloading data')
            X = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
                [
                    delayed(_load_image)(x, self.transforms) for x in X_paths
                ]
            )
            self.X = np.array(X)
            print('done')

        if y is not None:
            y = to_categorical(y, num_classes=self.num_classes)
        self.y = y

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.preload:
            X = self.X[indexes]
            if self.y is not None:
                y = self.y[indexes]
        else:
            X = np.empty((len(indexes), *self._dim, self.n_channels), dtype=np.float32)

            for i, idx in enumerate(indexes):
                X[i] = _load_image(self.X_paths[idx], self.transforms)

            if self.y is not None:
                y = self.y[indexes]

        if self.y is not None:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_model():
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg16.trainable = False  # set this to True to allow the full network to be trained
    inp = Input((224, 224, 3))
    x = vgg16(inp)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='linear', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(4096, activation='linear', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)        
    out = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inp, out)
    model.compile(
        loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
        metrics=['accuracy'])
    return model


class KerasImageClassifier():
    def __init__(self, workers=6, epochs=100):
        self.epochs = epochs
        self.workers = workers

    def fit(self, X, y):        
        X_train, X_valid, y_train, y_valid = \
            train_valid_split(X, y, valid_size=0.3, random_state=42)
        gen_train = DataGenerator(X_train, y_train, transforms=[transform_image])
        gen_valid = DataGenerator(X_valid, y_valid, transforms=[transform_image],
                                  shuffle=False)
        model = build_model()
        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.history = model.fit(
            gen_train,
            validation_data=gen_valid,
            use_multiprocessing=False,
            workers=self.workers, epochs=self.epochs,
            callbacks=[callback]
        )

        return self

    def predict_proba(self, X_paths):
        gen = DataGenerator(X_paths, transforms=[transform_image], shuffle=False)
        return self.model.predict(gen)


def get_estimator():
    return KerasImageClassifier()
