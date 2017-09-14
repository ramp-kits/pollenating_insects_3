from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam
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
	    samples_per_epoch=nb_train,
            nb_epoch=100,
            max_q_size=16,
            nb_worker=1,
            pickle_safe=True,
            validation_data=gen_valid,
            nb_val_samples=nb_valid,
            verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def _build_model(self):
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(3, 224, 224))
        # vgg16.trainable = False
        inp = Input((3, 224, 224))
        x = vgg16(inp)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='linear', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(4096, activation='linear', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)        
        out = Dense(403, activation='softmax', name='predictions')(x)
        model = Model(inp, out)
	NUM_GPU = 1 # or the number of GPUs available on your machine
	gpu_list = ['gpu(%d)' % i for i in range(NUM_GPU)]
        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
            metrics=['accuracy'],
            context=gpu_list)
        return model
