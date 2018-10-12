import sys, random, os
import numpy as np
from matplotlib import pyplot as plt
import pydot
import cv2
import quantize

NUM_EPOCHS = 2000

LR_D = 0.0004
LR_G = 0.001
BETA_1 = 0.8
EPSILON = 1e-4
ENC_WEIGHT = 200.0
BN_M = 0.8
DO_RATE = 0.25
NOISE_SIGMA = 0.15
CONTINUE_TRAIN = False
NUM_RAND_COMICS = 10
BATCH_SIZE = 16
PARAM_SIZE = 160
COMIC_PARAMS = 320

PREV_V = None
means = None
evals = None
evecs = None

def plotScores(scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	for s in scores:
		plt.plot(s)
	plt.xlabel('Epoch')
	loc = ('upper right' if on_top else 'lower right')
	plt.legend(['Dis', 'Gen', 'Enc'], loc=loc)
	plt.draw()
	plt.savefig(fname)

def save_config():
	with open('config.txt', 'w') as fout:
		fout.write('LR_D:        ' + str(LR_D) + '\n')
		fout.write('LR_G:        ' + str(LR_G) + '\n')
		fout.write('BETA_1:      ' + str(BETA_1) + '\n')
		fout.write('BN_M:        ' + str(BN_M) + '\n')
		fout.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
		fout.write('DO_RATE:     ' + str(DO_RATE) + '\n')
		fout.write('NOISE_SIGMA: ' + str(NOISE_SIGMA) + '\n')
		fout.write('EPSILON:     ' + str(EPSILON) + '\n')
		fout.write('ENC_WEIGHT:  ' + str(ENC_WEIGHT) + '\n')
		fout.write('optimizer_d: ' + type(d_optimizer).__name__ + '\n')
		fout.write('optimizer_g: ' + type(g_optimizer).__name__ + '\n')

def to_comic(fname, x):
	img = (x * 255.0).astype(np.uint8)
	if len(img.shape) == 4:
		img = np.concatenate(img, axis=2)
	img = np.transpose(img[::-1], (1, 2, 0))
	cv2.imwrite(fname, img)
	if fname == 'rand0.png':
		cv2.imwrite('rand0/r' + str(iters) + '.png', img)

def make_rand_comics(write_dir, rand_vecs):
	y_comics = generator.predict(rand_vecs)
	for i in xrange(rand_vecs.shape[0]):
		to_comic('rand' + str(i) + '.png', y_comics[i])

def make_rand_comics_normalized(write_dir, rand_vecs):
	global PREV_V
	global means
	global evals
	global evecs
	x_enc = np.squeeze(encoder.predict(x_samples))
	
	means = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - means).T)
	u, s, evecs = np.linalg.svd(x_cov)
	evals = np.sqrt(s)
	
	# This step is not necessary, but it makes the random generated test
	# samples consistent between epochs so you can see the evolution of
	# the training better.
	#
	# Like square roots, each prinicpal component has 2 solutions that
	# represent opposing vector directions.  For each component, just
	# choose the direction that was closest to the last epoch.
	if PREV_V is not None:
		d = np.sum(PREV_V * evecs, axis=1)
		d = np.where(d > 0.0, 1.0, -1.0)
		evecs = evecs * np.expand_dims(d, axis=1)
	PREV_V = evecs

	print "Evals: ", evals[:6]
	
	np.save(write_dir + 'means.npy', means)
	np.save(write_dir + 'stds.npy', x_stds)
	np.save(write_dir + 'evals.npy', evals)
	np.save(write_dir + 'evecs.npy', evecs)

	x_vecs = means + np.dot(rand_vecs * evals, evecs)
	make_rand_comics(write_dir, x_vecs)
	
	title = ''
	if '/' in write_dir:
		title = 'Epoch: ' + write_dir.split('/')[-2][1:]
	
	plt.clf()
	plt.title(title)
	plt.bar(np.arange(evals.shape[0]), evals, align='center')
	plt.draw()
	plt.savefig(write_dir + 'evals.png')

	plt.clf()
	plt.title(title)
	plt.bar(np.arange(means.shape[0]), means, align='center')
	plt.draw()
	plt.savefig(write_dir + 'means.png')
	
	plt.clf()
	plt.title(title)
	plt.bar(np.arange(x_stds.shape[0]), x_stds, align='center')
	plt.draw()
	plt.savefig(write_dir + 'stds.png')
	
def save_models():
	discriminator.save('discriminator.h5')
	generator.save('generator.h5')
	encoder.save('encoder.h5')
	print "Saved"
		
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
z_test = np.random.normal(0.0, 1.0, (NUM_RAND_COMICS, PARAM_SIZE))

###################################
#  Load Dataset
###################################
print "Loading Data..."
y_samples = np.load('data/top10000.npy')
y_shape = y_samples.shape
num_samples = y_samples.shape[0]
x_samples = np.expand_dims(np.arange(num_samples), axis=1)
x_shape = x_samples.shape
z_shape = (PARAM_SIZE,)
print "Loaded " + str(num_samples) + " panels."

y_test = y_samples[0].astype(np.float32) / 255.0
x_test = np.copy(x_samples[0:1])

###################################
#  Create Model
###################################
if CONTINUE_TRAIN:
	print "Loading Discriminator..."
	discriminator = load_model('discriminator.h5')
	print "Loading Generator..."
	generator = load_model('generator.h5')
	print "Loading Encoder..."
	encoder = load_model('encoder.h5')
	print "Loading Vectors..."
	PREV_V = np.load('evecs.npy')
	z_test = np.load('rand.npy')
else:
	print "Building Discriminator..."
	input_shape = y_shape[1:]
	print (None,) + input_shape
	discriminator = Sequential()
	discriminator.add(GaussianNoise(NOISE_SIGMA, input_shape=input_shape))

	discriminator.add(TimeDistributed(Conv2D(40, (5,5), padding='same')))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1)))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape
	discriminator.add(TimeDistributed(MaxPooling2D(4)))
	print discriminator.output_shape
	
	discriminator.add(TimeDistributed(Conv2D(80, (5,5), padding='same')))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1)))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape
	discriminator.add(TimeDistributed(MaxPooling2D(4)))
	print discriminator.output_shape
	
	discriminator.add(TimeDistributed(Conv2D(120, (5,5), padding='same')))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1)))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape
	discriminator.add(TimeDistributed(MaxPooling2D(8)))
	print discriminator.output_shape

	discriminator.add(Flatten(data_format = 'channels_last'))
	print discriminator.output_shape
	
	discriminator.add(Dense(1, activation='sigmoid'))
	print discriminator.output_shape

	print "Building Generator..."
	generator = Sequential()
	input_shape = (PARAM_SIZE,)
	print (None,) + input_shape

	generator.add(Dense(600, input_shape=input_shape))
	generator.add(LeakyReLU(0.2))
	generator.add(BatchNormalization(momentum=BN_M))
	print generator.output_shape
	
	generator.add(Dense(y_shape[1] * COMIC_PARAMS))
	generator.add(LeakyReLU(0.2))
	print generator.output_shape
	generator.add(Reshape((y_shape[1], COMIC_PARAMS)))
	generator.add(TimeDistributed(BatchNormalization(momentum=BN_M)))
	print generator.output_shape

	generator.add(TimeDistributed(Dense(200*4*4)))
	print generator.output_shape

	generator.add(Reshape((y_shape[1], 200,4,4)))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(TimeDistributed(Conv2DTranspose(200, (5,5), strides=(2,2), padding='same')))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(TimeDistributed(Conv2DTranspose(160, (5,5), strides=(2,2), padding='same')))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(TimeDistributed(Conv2DTranspose(120, (5,5), strides=(2,2), padding='same')))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(TimeDistributed(Conv2DTranspose(80, (5,5), strides=(2,2), padding='same')))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(TimeDistributed(Conv2DTranspose(40, (5,5), strides=(2,2), padding='same')))
	generator.add(LeakyReLU(0.2))
	if DO_RATE > 0:
		generator.add(Dropout(DO_RATE))
	#generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape
	
	generator.add(TimeDistributed(Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation='sigmoid')))
	print generator.output_shape

	print "Building Encoder..."
	encoder = Sequential()
	print (None, num_samples)
	encoder.add(Embedding(num_samples, PARAM_SIZE, input_length=1, embeddings_initializer=RandomNormal(stddev=1e-4)))
	encoder.add(Flatten(data_format = 'channels_last'))
	print encoder.output_shape
	
print "Building GANN..."
d_optimizer = Adam(lr=LR_D, beta_1=BETA_1, epsilon=EPSILON)
g_optimizer = Adam(lr=LR_G, beta_1=BETA_1, epsilon=EPSILON)

discriminator.trainable = True
generator.trainable = False
encoder.trainable = False
d_in_real = Input(shape=y_shape[1:])
d_in_fake = Input(shape=x_shape[1:])
d_fake = generator(encoder(d_in_fake))
d_out_real = discriminator(d_in_real)
d_out_real = Activation('linear', name='d_out_real')(d_out_real)
d_out_fake = discriminator(d_fake)
d_out_fake = Activation('linear', name='d_out_fake')(d_out_fake)
dis_model = Model(inputs=[d_in_real, d_in_fake], outputs=[d_out_real, d_out_fake])
dis_model.compile(
	optimizer=d_optimizer,
	loss={'d_out_real':'binary_crossentropy', 'd_out_fake':'binary_crossentropy'},
	loss_weights={'d_out_real':1.0, 'd_out_fake':1.0})

discriminator.trainable = False
generator.trainable = True
encoder.trainable = True
g_in = Input(shape=x_shape[1:])
g_enc = encoder(g_in)
g_out_img = generator(g_enc)
g_out_img = Activation('linear', name='g_out_img')(g_out_img)
g_out_dis = discriminator(g_out_img)
g_out_dis = Activation('linear', name='g_out_dis')(g_out_dis)
gen_dis_model = Model(inputs=[g_in], outputs=[g_out_img, g_out_dis])
gen_dis_model.compile(
	optimizer=g_optimizer,
	loss={'g_out_img':'mse', 'g_out_dis':'binary_crossentropy'},
	loss_weights={'g_out_img':ENC_WEIGHT, 'g_out_dis':1.0})
	
plot_model(gen_dis_model, to_file='generator.png', show_shapes=True)
plot_model(dis_model, to_file='discriminator.png', show_shapes=True)

###################################
#  Train
###################################
	
np.save('rand.npy', z_test)
to_comic('gt.png', y_test)
save_models()

print "Training..."
save_config()
generator_loss = []
discriminator_loss = []
encoder_loss = []

ones = np.ones((num_samples,), dtype=np.float32)
zeros = np.zeros((num_samples,), dtype=np.float32)

iters = 0
make_rand_comics_normalized('', z_test)

for iters in xrange(NUM_EPOCHS):
	loss_d = 0.0
	loss_g = 0.0
	loss_e = 0.0
	num_d = 0
	num_g = 0
	num_e = 0

	ratio_g = 1
	np.random.shuffle(x_samples)
	for i in xrange(0, num_samples/BATCH_SIZE):
		if i % ratio_g == 0:
			#Make samples
			j = i / ratio_g
			x_batch1 = x_samples[j*BATCH_SIZE:(j + 1)*BATCH_SIZE]
			y_batch1 = y_samples[x_batch1[:,0]].astype(np.float32) / 255.0
			
			ones = np.ones((BATCH_SIZE,), dtype=np.float32)
			zeros = np.zeros((BATCH_SIZE,), dtype=np.float32)

			losses = dis_model.train_on_batch([y_batch1, x_batch1], [ones, zeros])
			names = dis_model.metrics_names
			loss_d += losses[names.index('d_out_real_loss')]
			loss_d += losses[names.index('d_out_fake_loss')]
			num_d += 2


		x_batch2 = x_samples[i*BATCH_SIZE:(i + 1)*BATCH_SIZE]
		y_batch2 = y_samples[x_batch2[:,0]].astype(np.float32) / 255.0
		
		losses = gen_dis_model.train_on_batch([x_batch2], [y_batch2, ones])
		names = gen_dis_model.metrics_names
		loss_e += losses[names.index('g_out_img_loss')]
		loss_g += losses[names.index('g_out_dis_loss')]
		num_e += 1
		num_g += 1

		progress = (i * 100)*BATCH_SIZE / num_samples
		sys.stdout.write(
			str(progress) + "%" +
			"  D:" + str(loss_d / num_d) +
			"  G:" + str(loss_g / num_g) +
			"  E:" + str(loss_e / num_e) + "        ")
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('\n')
	
	discriminator_loss.append(loss_d / num_d)
	generator_loss.append(loss_g / num_g)
	encoder_loss.append(loss_e * 10.0 / num_e)

	plotScores([discriminator_loss, generator_loss, encoder_loss], 'Scores.png')

	save_models()

	#Generate some random comics
	y_enc = encoder.predict(x_test, batch_size=1)
	y_comic = generator.predict(y_enc, batch_size=1)[0]
	to_comic('test.png', y_comic)
	make_rand_comics_normalized('', z_test)

print "Done"
