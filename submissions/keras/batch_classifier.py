from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam
from rampwf.workflows.image_classifier import get_nb_minibatches


class BatchClassifier(object):
    def __init__(self):
        self.model = self._build_model()

    def fit(self, gen_builder):
        batch_size = 32
        gen_train, gen_valid, nb_train, nb_valid =\
            gen_builder.get_train_valid_generators(
                batch_size=batch_size, valid_ratio=0.1)
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=1,
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, batch_size),
            verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def _build_model(self):
        # This is VGG16
        inp = Input((3, 64, 64))
        # Block 1
        x = Conv2D(
            64, (3, 3), activation='relu', padding='same',
            name='block1_conv1')(inp)
        x = Conv2D(
            64, (3, 3), activation='relu', padding='same',
            name='block1_conv2')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block1_pool')(x)
        # Block 2
        x = Conv2D(
            128, (3, 3), activation='relu', padding='same',
            name='block2_conv1')(x)
        x = Conv2D(
            128, (3, 3), activation='relu', padding='same',
            name='block2_conv2')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block2_pool')(x)
        # Block 3
        x = Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv1')(x)
        x = Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv2')(x)
        x = Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv4')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block3_pool')(x)
        # Block 4
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block4_conv1')(x)
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block4_conv3')(x)
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block4_conv4')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block4_pool')(x)
        # Block 5
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block5_conv1')(x)
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block5_conv3')(x)
        x = Conv2D(
            512, (3, 3), activation='relu', padding='same',
            name='block5_conv4')(x)
        x = MaxPooling2D(
            (2, 2), strides=(2, 2),
            name='block5_pool')(x)
        # dense
        x = Flatten(name='flatten')(x)
        x = Dense(
            4096, activation='relu',
            name='fc1')(x)
        x = Dense(
            4096, activation='relu',
            name='fc2')(x)
        out = Dense(
            403, activation='softmax',
            name='predictions')(x)
        model = Model(inp, out)
        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])
        return model
