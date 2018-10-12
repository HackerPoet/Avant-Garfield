import sys, random, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2

DATA_SET = 'data/top10000.npy'
NUM_EPOCHS = 2000
LR = 0.0004
CONTINUE_TRAIN = False
USE_EMBEDDING = True
USE_MIRROR = False
NUM_RAND_COMICS = 10
DO_RATE = 0.0
BN_M = 0.8
BATCH_SIZE = 120
PARAM_SIZE_COMIC = 800
PARAM_SIZE_PANEL = 400
PREV_V = None

def plotScores(scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.xlabel('Epoch')
	loc = ('upper right' if on_top else 'lower right')
	plt.draw()
	plt.savefig(fname)

def save_config():
	with open('config.txt', 'w') as fout:
		fout.write('LR:          ' + str(LR) + '\n')
		fout.write('BN_M:        ' + str(BN_M) + '\n')
		fout.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
		fout.write('DO_RATE:     ' + str(DO_RATE) + '\n')
		fout.write('optimizer:   ' + type(model.optimizer).__name__ + '\n')

def as_float(y):
	return y.astype(np.float32) / 255.0
	
###################################
#  Load Keras
###################################
print "Loading Keras..."
import os, math
os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
import keras
print "Keras Version: " + keras.__version__
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector, ActivityRegularization, TimeDistributed, Lambda, LeakyReLU
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.initializers import RandomNormal
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import Sequence, plot_model
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')

#Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)

class Generator(Sequence):
    # Note, training set is too big to fit in memory as floating point.
	# Only convert to float when creating a batch.
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = as_float(self.y[inds])
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class GeneratorAE(Sequence):
    # Note, training set is too big to fit in memory as floating point.
	# Only convert to float when creating a batch.
    def __init__(self, y_set, batch_size=256):
        self.y =  y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.y.shape[0])

    def __len__(self):
        return math.ceil(self.y.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = as_float(self.y[inds])
        return batch_y, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

###################################
#  Load Dataset
###################################
print "Loading Data..."
y_samples = np.load(DATA_SET)
if USE_MIRROR:
	y_samples = np.concatenate([y_samples, y_samples[:,:,:,:,::-1]], axis=0)
y_shape = y_samples.shape
num_samples = y_samples.shape[0]
num_panels = y_samples.shape[1]
num_colors = y_samples.shape[2]
if USE_EMBEDDING:
	x_samples = np.expand_dims(np.arange(num_samples), axis=1)
	x_shape = x_samples.shape
print "Loaded " + str(num_samples) + " comics."

y_test = as_float(y_samples[44:45])
if USE_EMBEDDING:
	x_test = x_samples[44:45]

###################################
#  Create Model
###################################
if CONTINUE_TRAIN:
	print "Loading Model..."
	model = load_model('model.h5')
else:
	print "Building Model..."

	if USE_EMBEDDING:
		x_in = Input(shape=x_shape[1:])
		print (None,) + x_shape[1:]
		x = Dense(PARAM_SIZE_COMIC, use_bias=False, kernel_initializer=RandomNormal(stddev=1e-4))(x_in)
		x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
		print K.int_shape(x)
	else:
		x_in = Input(shape=y_shape[1:])
		print (None,) + y_shape[1:]
		x = TimeDistributed(Conv2D(40, (5,5), strides=(2,2), padding='same'))(x_in)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)
		
		x = TimeDistributed(Conv2D(80, (5,5), strides=(2,2), padding='same'))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)
		
		x = TimeDistributed(Conv2D(120, (5,5), strides=(2,2), padding='same'))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)

		x = TimeDistributed(Conv2D(160, (5,5), strides=(2,2), padding='same'))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)
		
		x = TimeDistributed(Conv2D(200, (5,5), strides=(2,2), padding='same'))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)
		
		x = TimeDistributed(Conv2D(200, (5,5)))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
		print K.int_shape(x)
		
		x = Reshape((num_panels, 200*4*4))(x)
		print K.int_shape(x)
		x = TimeDistributed(Dense(PARAM_SIZE_PANEL))(x)
		x = LeakyReLU(0.2)(x)
		x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
		print K.int_shape(x)

		x = Flatten()(x)
		print K.int_shape(x)
		x = Dense(PARAM_SIZE_COMIC)(x)
		x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
		print K.int_shape(x)

	x = Dense(1200, name='encoder')(x)
	x = LeakyReLU(0.2)(x)
	#x = BatchNormalization(momentum=BN_M)(x)
	print K.int_shape(x)
		
	x = Dense(num_panels * PARAM_SIZE_PANEL)(x)
	x = LeakyReLU(0.2)(x)
	#x = BatchNormalization(momentum=BN_M)(x)
	print K.int_shape(x)
	x = Reshape((num_panels, PARAM_SIZE_PANEL))(x)
	print K.int_shape(x)

	x = TimeDistributed(Dense(1600))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	print K.int_shape(x)
	
	x = TimeDistributed(Dense(240*4*4))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	print K.int_shape(x)
	x = Reshape((num_panels, 240, 4, 4))(x)
	print K.int_shape(x)
	
	x = TimeDistributed(Conv2DTranspose(200, (5,5)))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
	print K.int_shape(x)

	x = TimeDistributed(Conv2DTranspose(160, (5,5), strides=(2,2), padding='same'))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
	print K.int_shape(x)

	x = TimeDistributed(Conv2DTranspose(120, (5,5), strides=(2,2), padding='same'))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
	print K.int_shape(x)

	x = TimeDistributed(Conv2DTranspose(80, (5,5), strides=(2,2), padding='same'))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
	print K.int_shape(x)

	x = TimeDistributed(Conv2DTranspose(40, (5,5), strides=(2,2), padding='same'))(x)
	x = LeakyReLU(0.2)(x)
	#x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
	print K.int_shape(x)
	
	x = TimeDistributed(Conv2DTranspose(num_colors, (5,5), strides=(2,2), padding='same', activation='sigmoid'))(x)
	print K.int_shape(x)
	
	model = Model(x_in, x)
	model.compile(optimizer=Adam(lr=LR), loss='mse')

	plot_model(model, to_file='model.png', show_shapes=True)
	model.summary()

###################################
#  Train
###################################
print "Compiling SubModels..."
func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc = Model(inputs=model.input, outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (NUM_RAND_COMICS, PARAM_SIZE_COMIC))
np.save('rand.npy', rand_vecs)

def to_comic(fname, x):
	img = (x * 255.0).astype(np.uint8)
	img = np.concatenate(img, axis=2)
	img = np.transpose(img[::-1], (1, 2, 0))
	img = cv2.resize(img, (600, 180), interpolation = cv2.INTER_LINEAR)
	cv2.imwrite(fname, img)
	if fname == 'rand0.png':
		cv2.imwrite('rand0/r' + str(iter) + '.png', img)

def make_rand_comics(write_dir, rand_vecs):
	y_comics = func([rand_vecs, 0])[0]
	for i in xrange(rand_vecs.shape[0]):
		to_comic('rand' + str(i) + '.png', y_comics[i])

def make_rand_comics_normalized(write_dir, rand_vecs):
	global PREV_V
	x_enc = np.squeeze(enc.predict_generator(generator))
	
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	u, s, v = np.linalg.svd(x_cov)
	e = np.sqrt(s)

	# This step is not necessary, but it makes the random generated test
	# samples consistent between epochs so you can see the evolution of
	# the training better.
	#
	# Like square roots, each prinicpal component has 2 solutions that
	# represent opposing vector directions.  For each component, just
	# choose the direction that was closest to the last epoch.
	if PREV_V is not None:
		d = np.sum(PREV_V * v, axis=1)
		d = np.where(d > 0.0, 1.0, -1.0)
		v = v * np.expand_dims(d, axis=1)
	PREV_V = v
	
	print "Evals: ", e[:6]
	
	np.save(write_dir + 'means.npy', x_mean)
	np.save(write_dir + 'stds.npy', x_stds)
	np.save(write_dir + 'evals.npy', e)
	np.save(write_dir + 'evecs.npy', v)

	x_vecs = x_mean + np.dot(rand_vecs * e, v)
	make_rand_comics(write_dir, x_vecs)
	
	title = ''
	if '/' in write_dir:
		title = 'Epoch: ' + write_dir.split('/')[-2][1:]
	
	plt.clf()
	e[::-1].sort()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), e, align='center')
	plt.draw()
	plt.savefig(write_dir + 'evals.png')

	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_mean, align='center')
	plt.draw()
	plt.savefig(write_dir + 'means.png')
	
	plt.clf()
	plt.title(title)
	plt.bar(np.arange(e.shape[0]), x_stds, align='center')
	plt.draw()
	plt.savefig(write_dir + 'stds.png')

to_comic('gt.png', y_test[0])

print "Training..."
save_config()
train_loss = []
ofs = 0
if USE_EMBEDDING:
	generator = Generator(x_samples, y_samples, BATCH_SIZE)
else:
	generator = GeneratorAE(y_samples, BATCH_SIZE)

for iter in xrange(NUM_EPOCHS):
	history = model.fit_generator(generator)

	loss = history.history["loss"][-1]
	train_loss.append(loss)
	print "Train Loss: " + str(train_loss[-1])
	
	plotScores(train_loss, 'Scores.png', True)
	
	if iter % 3 == 0:
		model.save('model.h5')
		print "Saved"

		if USE_EMBEDDING:
			y_comic = model.predict(x_test, batch_size=BATCH_SIZE)[0]
		else:
			y_comic = model.predict(y_test, batch_size=BATCH_SIZE)[0]
		to_comic('test.png', y_comic)
		make_rand_comics_normalized('', rand_vecs)

print "Done"
