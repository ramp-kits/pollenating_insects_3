from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from rampwf.workflows.image_classifier import get_nb_minibatches


class BatchClassifier(object):
    def __init__(self):
        self.model = self._build_model()

    def fit(self, gen_builder):
        batch_size = 16
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
        vgg16 = VGG16(include_top=False, weights='imagenet')
        vgg16.trainable = False
        inp = vgg16.get_layer(name='input_1')
        hid = vgg16.get_layer(name='block3_conv3')
        vgg16_hid = Model(inp.input, hid.output)

        inp = Input((224, 224, 3))
        x = vgg16_hid(inp)
        x = Flatten(name='flatten')(x)
        x = Dense(200, activation='linear', name='fc')(x)
        x = BatchNormalization()(x)
        out = Dense(403, activation='softmax', name='predictions')(x)
        model = Model(inp, out)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.95), metrics=['accuracy'])
        return model

