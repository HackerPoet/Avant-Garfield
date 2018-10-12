import sys, random, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import quantize

MODEL_DIR = 'model.h5'
IS_FULL = True
BATCH_SIZE = 64
TOP_N = 50

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
from keras.utils import plot_model
from keras.activations import softmax
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')
	
#Fix the random seed so that training comparisons are easier to make
np.random.seed(0)
random.seed(0)

def as_float(y):
	return y.astype(np.float32) / 255.0

def to_comic(fname, x):
	img = (x * 255.0).astype(np.uint8)
	if len(img.shape) == 4:
		img = np.concatenate(img, axis=2)
	img = np.transpose(img[::-1], (1, 2, 0))
	cv2.imwrite(fname, img)

def save_comics(fname, samples):
	for i in xrange(samples.shape[0]):
		to_comic(fname + str(i) + '.png', samples[i])

def predict_full(ix, x_samples, model):
	if IS_FULL:
		return model.predict(x_samples[ix], batch_size=BATCH_SIZE)
	else:
		ix_full = np.empty((ix.shape[0]*3,), dtype=ix.dtype)
		ix_full[0::3] = ix*3
		ix_full[1::3] = ix*3 + 1
		ix_full[2::3] = ix*3 + 2
		y = model.predict(x_samples[ix_full], batch_size=BATCH_SIZE)
		return np.reshape(y, (3, y.shape[0]/3) + y.shape[1:])

###################################
print "Loading Data..."
y_samples = np.load('data/comics.npy')
orig_shape = y_samples.shape
if not IS_FULL:
	y_samples = y_samples.reshape((y_samples.shape[0] * y_samples.shape[1],) + y_samples.shape[2:])
y_shape = y_samples.shape
num_samples = y_shape[0]
x_samples = np.arange(num_samples)
print "Loaded " + str(num_samples) + " panels."

print "Loading Model..."
model = load_model(MODEL_DIR)

print "Evaluating model..."
z_scores = np.zeros((num_samples,))
for i in xrange(0, num_samples, BATCH_SIZE):
	x_batch = x_samples[i:i+BATCH_SIZE]
	y_batch = y_samples[i:i+BATCH_SIZE]
	y_batch = as_float(y_batch)
	batch_size = x_batch.shape[0]
	
	y_pred = model.predict(x_batch, batch_size=BATCH_SIZE)
	axis = (1,2,3,4) if IS_FULL else (1,2,3)
	z_scores[i:i+BATCH_SIZE] = np.mean(np.square(y_pred - y_batch), axis=axis)

print "Sorting Scores..."
if not IS_FULL:
	assert(z_scores.shape[0] % 3 == 0)
	z_scores = np.reshape(z_scores, (z_scores.shape[0]/3, 3))
	z_scores = np.amax(z_scores, axis=1)
	y_samples = np.reshape(y_samples, orig_shape)
ids = np.argsort(z_scores)
best = ids[:TOP_N]
worst = ids[-TOP_N:]
save_comics('BestWorst/best_gt', as_float(y_samples[best]))
save_comics('BestWorst/worst_gt', as_float(y_samples[worst]))
save_comics('BestWorst/best_pred', predict_full(best, x_samples, model))
save_comics('BestWorst/worst_pred', predict_full(worst, x_samples, model))

print "Saving..."
np.save('data/top5000.npy', y_samples[ids[:5000]])
np.save('data/top10000.npy', y_samples[ids[:10000]])

print "Done"
