import pygame
import random
import numpy as np
import cv2
import h5py

#User constants
device = "cpu"
model_dir = ''
is_gan = False
background_color = (210, 210, 210)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]
image_scale = 2
image_padding = 10
slider_w = 20
slider_h = 200
slider_px = 5
slider_py = 12
slider_cols = 51
slider_rows = 1
num_stds = 4

#Keras
print "Loading Keras..."
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.local import LocallyConnected2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

print "Loading Encoder..."
if is_gan:
	enc_model = load_model(model_dir + 'generator.h5')
	num_params = enc_model.layers[0].input_shape[1]
	comic_shape = enc_model.layers[-1].output_shape[1:]
else:
	enc_model = load_model(model_dir + 'model.h5')
	num_params = enc_model.get_layer('encoder').input_shape[1]
	comic_shape = enc_model.output_shape[1:]
	enc = K.function([enc_model.get_layer('encoder').input, K.learning_phase()],
					 [enc_model.layers[-1].output])
				 
print "Loading Statistics..."
means = np.load(model_dir + 'means.npy')
stds  = np.load(model_dir + 'stds.npy')
evals = np.load(model_dir + 'evals.npy')
evecs = np.load(model_dir + 'evecs.npy')

sort_inds = np.argsort(-evals)
evals = evals[sort_inds]
evecs = evecs[:,sort_inds]

if len(comic_shape) == 3:
	input_w = comic_shape[-1]
	input_h = comic_shape[-2]
else:
	input_w = comic_shape[0] * comic_shape[-1]
	input_h = comic_shape[-2]

#Derived constants
slider_w = slider_w + slider_px*2
slider_h = slider_h + slider_py*2
drawing_x = image_padding
drawing_y = image_padding
drawing_w = input_w * image_scale
drawing_h = input_h * image_scale
sliders_x = image_padding
sliders_y = drawing_y + drawing_h + image_padding
sliders_w = slider_w * slider_cols
sliders_h = slider_h * slider_rows
window_w = drawing_w + image_padding*2
window_h = drawing_h + image_padding*3 + sliders_h
num_ticks = num_stds*2 + 1

#Global variables
prev_mouse_pos = None
mouse_pressed = False
cur_slider_ix = 0
needs_update = True
cur_params = np.zeros((num_params,), dtype=np.float32)
cur_comic = np.zeros((3, input_h, input_w), dtype=np.uint8)
rgb_array = np.zeros((input_h, input_w, 3), dtype=np.uint8)

#Open a window
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((window_w, window_h))
comic_surface_mini = pygame.Surface((input_w, input_h))
comic_surface = screen.subsurface((drawing_x, drawing_y, drawing_w, drawing_h))
pygame.display.set_caption('Comic Editor - By <CodeParade>')
font = pygame.font.SysFont("monospace", 15)

def update_mouse_click(mouse_pos):
	global cur_slider_ix
	global mouse_pressed
	x = (mouse_pos[0] - sliders_x)
	y = (mouse_pos[1] - sliders_y)

	if x >= 0 and y >= 0 and x < sliders_w and y < sliders_h:
		slider_ix_w = x / slider_w
		slider_ix_h = y / slider_h

		cur_slider_ix = slider_ix_h * slider_cols + slider_ix_w
		mouse_pressed = True

def update_mouse_move(mouse_pos):
	global needs_update
	y = (mouse_pos[1] - sliders_y)

	if y >= 0 and y < sliders_h:
		slider_row_ix = cur_slider_ix / slider_cols
		slider_val = y - slider_row_ix * slider_h

		slider_val = min(max(slider_val, slider_py), slider_h - slider_py) - slider_py
		val = (float(slider_val) / (slider_h - slider_py*2) - 0.5) * (num_stds*2)
		cur_params[cur_slider_ix] = val
		
		needs_update = True

def draw_sliders():
	for row in xrange(slider_rows):
		for col in xrange(slider_cols):
			i = row*slider_cols + col
			slider_color = slider_colors[i % len(slider_colors)]

			x = sliders_x + col * slider_w
			y = sliders_y + row * slider_h

			cx = x + slider_w / 2
			cy_1 = y + slider_py
			cy_2 = y + slider_h - slider_py
			pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))
			
			py = y + int((cur_params[i] / (num_stds*2) + 0.5) * (slider_h - slider_py*2)) + slider_py
			pygame.draw.circle(screen, slider_color, (cx, py), slider_w/2 - slider_px)
			
			cx_1 = x + slider_px
			cx_2 = x + slider_w - slider_px
			for j in xrange(num_ticks):
				ly = int(y + slider_h*(0.5 + float(j-num_stds)/num_ticks))
				pygame.draw.line(screen, slider_color, (cx_1, ly), (cx_2, ly))

def draw_comic():
	pygame.surfarray.blit_array(comic_surface_mini, np.transpose(cur_comic, (2, 1, 0)))
	pygame.transform.scale(comic_surface_mini, (drawing_w, drawing_h), comic_surface)
	pygame.draw.rect(screen, (0,0,0), (drawing_x, drawing_y, drawing_w, drawing_h), 1)
	
#Main loop
running = True
while running:
	#Process events
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			break
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				prev_mouse_pos = pygame.mouse.get_pos()
				update_mouse_click(prev_mouse_pos)
				update_mouse_move(prev_mouse_pos)
			elif pygame.mouse.get_pressed()[2]:
				cur_params = np.zeros((num_params,), dtype=np.float32)
				needs_update = True
		elif event.type == pygame.MOUSEBUTTONUP:
			mouse_pressed = False
			prev_mouse_pos = None
		elif event.type == pygame.MOUSEMOTION and mouse_pressed:
			update_mouse_move(pygame.mouse.get_pos())
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_stds, num_stds)
				needs_update = True
			elif event.key == pygame.K_1:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_stds, num_stds)
				cur_params[20:] = 0.0
				needs_update = True
			elif event.key == pygame.K_2:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_stds, num_stds)
				cur_params[40:] = 0.0
				needs_update = True
			elif event.key == pygame.K_3:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_stds, num_stds)
				cur_params[60:] = 0.0
				needs_update = True
			elif event.key == pygame.K_4:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -num_stds, num_stds)
				cur_params[80:] = 0.0
				needs_update = True

	#Check if we need an update
	if needs_update:
		x = means + np.dot(cur_params * evals, evecs)
		#x = means + stds * cur_params
		x = np.expand_dims(x, axis=0)
		if is_gan:
			y = enc_model.predict(x)[0]
		else:
			y = enc([x, 0])[0][0]
		if len(y.shape) == 4:
			y = np.concatenate(y, axis=2)
		cur_comic[:] = (y * 255.0).astype(np.uint8)
		needs_update = False
	
	#Draw to the screen
	screen.fill(background_color)
	draw_comic()
	draw_sliders()
	
	#Flip the screen buffer
	pygame.display.flip()
	pygame.time.wait(10)
